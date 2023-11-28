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
from transformers import WhisperFeatureExtractor

from SingleModels.models.text import BertClassifier
from SingleModels.models.whisper import WhisperForEmotionClassification
from SingleModels.models.video import VideoClassification

import numpy as np
from numpy.random import choice
from torch.nn.utils.rnn import pad_sequence
from torch import nn


FEAT = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

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
    path_video = []
    target_timings = []
    context_timings_list = []

    if not must:
        video_context = [torch.empty((1, 1, 1, 1)), torch.empty((1, 1, 1, 1))]
    else:
        video_context = []
    speech_list_context_input_values = torch.empty((1))

    for input, label in batch:
        text = input[0]
        input_list.append(text["input_ids"].tolist()[0])
        text_mask.append(text["attention_mask"].tolist()[0])
        audio_path = input[1]
        vid_features = input[2]  # [6:] for debug
        if not must:
            path_video.append(vid_features[0])
            target_timings.append(vid_features[2])
            
            speech_list.append(FEAT(audio_path[1] , sampling_rate=16000).input_features[0])
            video_list.append(vid_features[1])
        else:
            path_video.append(vid_features[0])
            target_timings.append(vid_features[3])
            
            context_timings_list.append(audio_path[-1][0]) # The 0 is always context
            speech_list.append(FEAT(audio_path[1], sampling_rate=16000).input_features[0])
            speech_context.append(FEAT(audio_path[2], sampling_rate=16000).input_features[0])
            video_list.append(vid_features[1])
            video_context.append(vid_features[2])

        label_list.append(label)
    if must:
        speech_list_context_input_values = torch.Tensor(np.array(speech_context))
        
    batch_size = len(label_list)
    vid_mask = torch.zeros(batch_size, 1568).bool()
    idx = torch.arange(1568)
    idx_to_change = torch.Tensor(choice(idx, size=1056, replace=False))
    # Repeat this tensor batch_size times
    idx_to_change = idx_to_change.repeat(batch_size, 1).long()
    vid_mask[:, idx_to_change] = True
    
    text = {
        "input_ids": torch.Tensor(np.array(input_list)).type(torch.LongTensor),
        "text_attention_mask": torch.Tensor(np.array(text_mask)),
        "timings": target_timings,
    }

    audio_features = {
        "audio_features": torch.Tensor(np.array(speech_list)),
        "context_audio": speech_list_context_input_values,
        "context_timings_list" : context_timings_list
    }

    visual_embeds = {
        "video_path" : path_video,
        "video_embeds": torch.stack(video_list).permute(0, 2, 1, 3, 4),
        "video_context": torch.stack(video_context).permute(0, 2, 1, 3, 4),
        "video_mask": vid_mask,
    }
    return {**text , **audio_features , **visual_embeds}, torch.Tensor(np.array(label_list)).long()




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
        
        args['BertModel'] = 'roberta-large'
        
        self.must = False
        self.tiktok = False
        if "meld" in str(self.dataset).lower():
            dataset_name = "meld"
            path = [
            f"{dataset_name}_bert/ml07txx0/clean-sweep-6/best.pt",
            
            f"{dataset_name}_whisper/ddadfsl6/efficient-sweep-3/best.pt",
            
            f"{dataset_name}_video/qkcuxp10/fine-sweep-17/best.pt",
            ]
        elif "iemo" in str(self.dataset).lower():
            dataset_name = "iemo"
            path = [
            f"{dataset_name}_bert/ylhfsh3s//best.pt",
            
            f"{dataset_name}_whisper/xaz13i9h/proud-sweep-34/best.pt",
            
            f"{dataset_name}_video/myt3iav2/rose-sweep-69/best.pt",
            ]
        else:
            dataset_name = "mustard"
            self.must = True
            path = [
            f"{dataset_name}_bert/aoago6hk/woven-sweep-62/best.pt",
            
            f"{dataset_name}_whisper/qzyfm99j/astral-sweep-24/best.pt",
            
            f"{dataset_name}_video/apl9ufpx/crisp-sweep-51/best.pt",
            ]
            
        # self.f = h5py.File(f"../../data/{dataset_name}.final.seq_len.features.hdf5", "a", libver="latest", swmr=True)
        # try:
        #     self.f.swmr_mode = True
        # except:
        #     pass

        self.p = 0.75

        self.bert = BertClassifier(args)
        self.whisper = WhisperForEmotionClassification(args)
        self.videomae = VideoClassification(args)
        
        

        # Load in our finetuned models, exactly as they should be
        
        for i, model in enumerate([ self.bert , self.whisper, self.videomae]):
            print(f"Loading from {path[i]}, " , flush = True)
            ckpt =  torch.load(f"../../../TAV_Train/{path[i]}" , map_location=torch.device('cuda'))['model_state_dict']
            model.load_state_dict(ckpt)
        
        
        # Zero out all gradients. Might not need this anymore though
        # for i, model in enumerate([self.bert, self.whisper, self.videomae]):
        #     for param in model.base_model.parameters():
        #         param.requires_grad = False

        # Put the models onto eval mode
        self.bert.eval()
        self.whisper.eval()
        self.videomae.eval()

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

    def forward(
        self,
        input_ids,
        text_attention_mask,
        audio_features,
        context_audio,
        video_embeds,
        video_mask,
        video_context,
        video_path,
        timings,
        context_timings_list,
        check="train",
    ):
        # Transformer Time
        # last_hidden_text_state: torch.Tensor
        # text_outputs : torch.Tensor
        # aud_outputs: torch.Tensor
        # aud_context : torch.Tensor
        # vid_outputs: torch.Tensor
        # vid_context : torch.Tensor
        
        
        last_hidden_text_state, text_outputs = self.bert.bert(
            input_ids=input_ids, attention_mask=text_attention_mask, return_dict=False
        )
        
        # if self.must:
        #     self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0]}/text", data=last_hidden_text_state.cpu().detach().numpy())
        # else:
        #     self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text", data=last_hidden_text_state.cpu().detach().numpy())
        

        # aud_outputs = self.whisper.whisper.encoder(audio_features)[0][:,:512,:]
        
        
        # if self.must:
        #     aud_context = self.whisper.whisper.encoder(context_audio)[0][:,:512,:]
        #     del context_audio
        #     new_aud_context = torch.zeros_like(aud_context[:,:512,:]) # Cut it to be this, now assign it
        #     for i , row in enumerate(aud_context):
        #         if context_timings_list[i] == None:
        #             new_aud_context[i] = row[:512] # If less then 10.24 seconds then take first 10.24 seconds
                    
        #         elif context_timings_list[i][1] - context_timings_list[i][0] < 10.24:
        #             new_aud_context[i] = row[:512] # If less then 10.24 seconds then take first 10.24 seconds
                    
        #         else: # Take the last 10.24 seconds
        #             new_aud_context[i] = row[-512:]
        #     del aud_context
                
        #     self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/audio_context", data=aud_context.cpu().detach().numpy())
        #     self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/audio", data=aud_outputs.cpu().detach().numpy())
        #     aud_outputs = (aud_outputs * self.p + new_aud_context * (1 - self.p)) / 2
            
        # else:
        #     self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/audio", data=aud_outputs.cpu().detach().numpy())
            

        # vid_outputs = self.videomae(video_embeds, bool_masked_pos = video_mask)[0]  
        

        # if self.must:
        #     vid_context = self.videomae(video_context, bool_masked_pos = video_mask)[0]
            
        #     self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/video_context", data=vid_context.cpu().detach().numpy())
        #     self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/video", data=vid_outputs.cpu().detach().numpy())
        # else:
        #     self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/video", data=vid_outputs.cpu().detach().numpy())
        
        tav = self.bert.linear1(text_outputs)
        # aud_outputs = self.whisper.linear1(aud_outputs.mean(dim=1))
        
        # vid_outputs = self.videomae.linear1(vid_outputs[:,0])
        
        # tav = (text_outputs + aud_outputs + vid_outputs) / 3
        
        return tav

