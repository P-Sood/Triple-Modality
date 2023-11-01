from copy import deepcopy
from glob import glob
from transformers import logging
import h5py
import pdb

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")
import torch
from torch import nn
from transformers import VideoMAEModel, AutoModel, AutoProcessor, AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

import numpy as np
from numpy.random import choice
from torch.nn.utils.rnn import pad_sequence
from torch import nn

def collate_batch(batch, must):  # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    speech_context = []
    speech_list = []
    label_list = []
    input_list = []
    text_mask = []
    path_audio = []
    path_video = []
    target_timings = []

    if not must:
        video_context = [torch.empty((1, 1, 1, 1)), torch.empty((1, 1, 1, 1))]
    else:
        video_context = []

    speech_list_context_input_values = torch.empty((1))
    speech_list_mask = None
    vid_mask = None

    for data, label in batch:
        text = data[0]
        data_list.append(text["data_ids"].tolist()[0])
        text_mask.append(text["attention_mask"].tolist()[0])
        audio_path = data[1]
        vid_features = data[2]  # [6:] for debug

        if not must:
            path_audio.append(audio_path[0])
            path_video.append(vid_features[0])
            target_timings.append(vid_features[2])
            
            speech_list.append(audio_path[1])
            video_list.append(vid_features[1])
        else:
            path_audio.append(audio_path[0])
            path_video.append(vid_features[0])
            target_timings.append(vid_features[3])
            
            speech_list.append(audio_path[1])
            speech_context.append(audio_path[2])
            video_list.append(vid_features[1])
            video_context.append(vid_features[2])

        label_list.append(label)
    batch_size = len(label_list)

    vid_mask = torch.randint(
        -13, 2, (batch_size, 1568)
    )  # 8*14*14 = 1568: Seq len for all videos processed w VideoMAE 
    vid_mask[vid_mask > 0] = 0
    vid_mask = vid_mask.bool()  # RAndom mask over values
    x = torch.count_nonzero(vid_mask.int()).item()
    rem = (1568 * batch_size - x) % batch_size

    # What does this if statement do? I.e., what is "rem"?
    if rem != 0:
        idx = torch.where(vid_mask.view(-1) == 0)[0]  # Flatten and get 0-indices
        num_to_change = rem  # as follows from example abow
        idx_to_change = choice(idx, size=num_to_change, replace=False)
        vid_mask.view(-1)[idx_to_change] = 1

    speech_list_input_values = pad_sequence(
        speech_list, batch_first=True, padding_value=0
    )
    del speech_list

    if must:
        speech_list_context_input_values = pad_sequence(
            speech_context, batch_first=True, padding_value=0
        )
        del speech_context

    text = {
        "input_ids": torch.Tensor(np.array(input_list)).type(torch.LongTensor),
        "attention_mask": torch.Tensor(np.array(text_mask)),
        "timings": target_timings,
    }

    audio_features = {
        "audio_path" : path_audio,
        "audio_features": speech_list_input_values,
        "attention_mask": speech_list_mask,
        "audio_context": speech_list_context_input_values,
    }

    visual_embeds = {
        "video_path" : path_video,
        "visual_embeds": torch.stack(video_list).permute(0, 2, 1, 3, 4),
        "attention_mask": vid_mask,
        "visual_context": torch.stack(video_context).permute(0, 2, 1, 3, 4),
    }
    return [text, audio_features, visual_embeds], torch.Tensor(np.array(label_list))

class TAVEncoder(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(TAVForMAE_HDF5, self).__init__()
        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.learn_PosEmbeddings = args["learn_PosEmbeddings"]
        self.num_layers = args["num_layers"]
        self.dataset = args["dataset"]
        self.sota = args["sota"]

        if "meld" in str(self.dataset.lower()):
            dataset_name = "meld"
        elif "iemo" in str(self.dataset).lower():
            dataset_name = "iemo"
        elif "tiktok" in str(self.dataset).lower():
            dataset_name = "tiktok"
            self.tiktok = True
        else:
            dataset_name = "mustard"
            self.must = True
        
        self.f = h5py.File(f"../../data/{dataset_name}.features.hdf5", "a", liver="latest", swmr=True)
        self.f.swmr_mode = True

        self.p = 0.6  # Document this.

        # Load pre-trained models.
        if self.must:
            self.bert = AutoModel.from_pretrained("jkhan447/sarcasm-detection-RoBerta-base-CR")
            self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        elif self.tiktok:
            self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
            self.wav2vec2 = AutoModel.from_pretrained("justin1983/wav2vec2-xlsr-multilingual-56-finetuned-amd")
        else:
            self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
            self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        for model in [self.bert, self.wav2vec2, self.videomae]:
            for param in model.base_model.parameters():
                param.requires_grad = False
        self.linear1 = nn.Linear(768, self.output_dim)

    def forward(
        self,
        input_ids,
        text_attention_mask,
        audio_features,
        context_audio,
        audio_path,
        video_embeds,
        video_context,
        video_path,
        visual_mask,
        timings,
        check="train",
    ):
        # print("video_path", video_path , flush=True)
        # Transformer Time
        _, text_outputs = self.bert(
            input_ids=input_ids, attention_mask=text_attention_mask, return_dict=False
        )
        
        # Create HDF5 datasets
        if self.must:
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0]}/text", data=text_outputs.cpu().detach().numpy())
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text", data=text_outputs.cpu().detach().numpy())

        # Get audio features
        aud_outputs = self.wav2vec2(audio_features)[0]
        aud_outputs = torch.mean(aud_outputs, dim=1)

        # Delete information to take up less GPU mem
        del _
        del input_ids
        del text_attention_mask
        del audio_features
        
        if self.must:
            aud_context = self.wav2vec2(context_audio)[0]
            aud_context = torch.mean(aud_context, dim=1)
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/audio_context", data=aud_context.cpu().detach().numpy())
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/audio", data=aud_outputs.cpu().detach().numpy())
            del aud_context
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/audio", data=aud_outputs.cpu().detach().numpy())

        vid_outputs = self.videomae(video_embeds, visual_mask)[0]  
        vid_outputs = torch.mean(vid_outputs, dim=1)
        del video_embeds

        if self.must:
            vid_context = self.videomae(video_context, visual_mask)[0]
            vid_context = torch.mean(vid_context, dim=1)
            del video_context
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/video_context", data=vid_context.cpu().detach().numpy())
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/video", data=vid_outputs.cpu().detach().numpy())
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/video", data=vid_outputs.cpu().detach().numpy())
            
        tav = self.linear1(text_outputs)
        
        return tav  # returns [batch_size,output_dim]
