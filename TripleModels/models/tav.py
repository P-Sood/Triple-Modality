from copy import deepcopy
from glob import glob
from transformers import logging

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

    for input, label in batch:
        text = input[0]
        input_list.append(text["input_ids"].tolist()[0])
        text_mask.append(text["attention_mask"].tolist()[0])
        audio_path = input[1]
        vid_features = input[2]  # [6:] for debug
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
    )  # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it,
    vid_mask[vid_mask > 0] = 0
    vid_mask = vid_mask.bool()
    # now we have a random mask over values so it can generalize better
    x = torch.count_nonzero(vid_mask.int()).item()
    rem = (1568 * batch_size - x) % batch_size
    if rem != 0:
        idx = torch.where(vid_mask.view(-1) == 0)[
            0
        ]  # get all indicies of 0 in flat tensor
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


class TAVForMAE(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(TAVForMAE, self).__init__()
        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.learn_PosEmbeddings = args["learn_PosEmbeddings"]
        self.num_layers = args["num_layers"]
        self.dataset = args["dataset"]
        self.sota = args["sota"]
        
        print(f"Using {self.num_layers} layers \nUsing sota = {self.sota}" , flush=True)

        self.must = True if "must" in str(self.dataset).lower() else False
        self.tiktok = True if "tiktok" in str(self.dataset).lower() else False
        self.p = 0.6

        if self.must:
            self.bert = AutoModel.from_pretrained("jkhan447/sarcasm-detection-RoBerta-base-CR")
        elif self.tiktok:
            self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
        else:
            self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.test_ctr = 1
        self.train_ctr = 1
        
        if self.must:
            self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        elif self.tiktok:
            self.wav2vec2 = AutoModel.from_pretrained("justin1983/wav2vec2-xlsr-multilingual-56-finetuned-amd")
        else:
            self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")




        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        for model in [self.bert, self.wav2vec2, self.videomae]:
            for param in model.base_model.parameters():
                param.requires_grad = False

        # Everything before this line is unlearnable, everything after is what we are focused on
        self.wav_2_768_2 = nn.Linear(1024, 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)
        self.aud_norm = nn.LayerNorm(1024)

        self.bert_norm = nn.LayerNorm(768)
        self.vid_norm = nn.LayerNorm(768)

        self.aud_text_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=768, num_heads=8)
                for _ in range(self.num_layers)
            ]
        )
        self.vid_text_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=768, num_heads=8)
                for _ in range(self.num_layers)
            ]
        )
        self.fusion_layers = nn.ModuleList(
            [
                nn.Linear(768*2, 768)
                for _ in range(self.num_layers)
            ]
        )

        self.dropout = nn.Dropout(self.dropout)
        if self.sota:
            self.linear1 = nn.Linear(768 * 3, 768 * 2)
        else:
            self.linear1 = nn.Linear(768 * 4, 768 * 2)
            
        self.linear2 = nn.Linear(768 * 2, self.output_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids,
        text_attention_mask,
        audio_features,
        context_audio,
        video_embeds,
        video_context,
        visual_mask,
        check="train",
    ):
        # Transformer Time
        _, text_outputs = self.bert(
            input_ids=input_ids, attention_mask=text_attention_mask, return_dict=False
        )

        del _
        del input_ids
        del text_attention_mask
        text_outputs = self.bert_norm(text_outputs)

        aud_outputs = self.wav2vec2(audio_features)[0]
        del audio_features
        aud_outputs = torch.mean(aud_outputs, dim=1)
        aud_outputs = self.aud_norm(aud_outputs)
        if self.must:
            aud_context = self.wav2vec2(context_audio)[0]
            del context_audio
            aud_context = torch.mean(aud_context, dim=1)
            aud_context = self.aud_norm(aud_context)
            aud_outputs = (aud_outputs * self.p + aud_context * (1 - self.p)) / 2
            del aud_context

        aud_outputs = self.wav_2_768_2(aud_outputs)
        vid_outputs = self.videomae(video_embeds, visual_mask)[
            0
        ]  # Now it has 2 dimensions
        del video_embeds

        vid_outputs = torch.mean(vid_outputs, dim=1)  # Now it has 2 dimensions
        vid_outputs = self.vid_norm(vid_outputs)

        if self.must:
            vid_context = self.videomae(video_context, visual_mask)[0]
            del video_context
            vid_context = torch.mean(vid_context, dim=1)  # Now it has 2 dimensions
            vid_context = self.vid_norm(vid_context)
            vid_outputs = (vid_outputs * self.p + vid_context * (1 - self.p)) / 2

        del visual_mask

        # Now we have to concat all the outputs
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f"Free mem: {f}", flush=True)
       
        if self.sota:
            for i in range(self.num_layers):
                Ffusion1 = text_outputs
                Ffusion2 = text_outputs
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                fusion_layer = self.fusion_layers[i]
                Ffusion1, _ = aud_text_layer(Ffusion1, Ffusion1, aud_outputs)
                Ffusion2, _ = vid_text_layer(Ffusion2, Ffusion2, vid_outputs)
                text_outputs = fusion_layer(torch.cat([Ffusion1, Ffusion2], dim=1))
            tav = torch.cat([text_outputs, aud_outputs, vid_outputs], dim=1)
        else:
            Ffusion1 = text_outputs
            Ffusion2 = text_outputs
            for i in range(self.num_layers):
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                Ffusion1, _ = aud_text_layer(Ffusion1, aud_outputs, aud_outputs)
                Ffusion2, _ = vid_text_layer(Ffusion2, vid_outputs, vid_outputs)
            tav = torch.cat([Ffusion1, Ffusion2, aud_outputs, vid_outputs], dim=1)

        del text_outputs
        del aud_outputs
        del vid_outputs

        # Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)
        tav = self.relu(tav)
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear2(tav)

        return tav  # returns [batch_size,output_dim]

import h5py
import pdb
class TAVForMAE_HDF5(nn.Module):
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
        
        if "meld" in str(self.dataset).lower():
            self.f = h5py.File("../../data/meld_features.hdf5", "a", libver="latest", swmr=True)
        elif "iemo" in str(self.dataset).lower():
            self.f = h5py.File("../../data/iemo_features.hdf5", "a", libver="latest", swmr=True)
        elif "tiktok" in str(self.dataset).lower():
            self.f = h5py.File("../../data/tiktok_features.hdf5", "a", libver="latest", swmr=True)
        else:
            self.f = h5py.File("../../data/mustard_features.hdf5", "a", libver="latest", swmr=True)
        self.f.swmr_mode = True
        print(f"Using {self.num_layers} layers \nUsing sota = {self.sota}" , flush=True)

        self.must = True if "must" in str(self.dataset).lower() else False
        self.tiktok = True if "tiktok" in str(self.dataset).lower() else False
        self.p = 0.6

        if self.must:
            self.bert = AutoModel.from_pretrained("jkhan447/sarcasm-detection-RoBerta-base-CR")
        elif self.tiktok:
            self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
        else:
            self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.test_ctr = 1
        self.train_ctr = 1
        
        if self.must:
            self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        elif self.tiktok:
            self.wav2vec2 = AutoModel.from_pretrained("justin1983/wav2vec2-xlsr-multilingual-56-finetuned-amd")
        else:
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
        
        if self.must:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text", data=text_outputs)
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text", data=text_outputs)
        del _
        del input_ids
        del text_attention_mask

        aud_outputs = self.wav2vec2(audio_features)[0]
        aud_outputs = torch.mean(aud_outputs, dim=1)
        del audio_features
        
        if self.must:
            aud_context = self.wav2vec2(context_audio)[0]
            aud_context = torch.mean(aud_context, dim=1)
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/audio_context", data=aud_context)
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/audio", data=aud_outputs)
            del aud_context
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/audio", data=aud_outputs)
            

        vid_outputs = self.videomae(video_embeds, visual_mask)[0]  
        vid_outputs = torch.mean(vid_outputs, dim=1)
        del video_embeds

        if self.must:
            vid_context = self.videomae(video_context, visual_mask)[0]
            vid_context = torch.mean(vid_context, dim=1)
            del video_context
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/video_context", data=vid_context)
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/video", data=vid_outputs)
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/video", data=vid_outputs)
            
        tav = self.linear1(text_outputs)
        
        return tav  # returns [batch_size,output_dim]
