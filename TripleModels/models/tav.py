from copy import deepcopy
from glob import glob
from typing import OrderedDict
from transformers import logging
import h5py
import pdb
from transformers import AutoModel, AutoModelForSequenceClassification 

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

    speech_list = []
    label_list = []
    input_list = []
    text_mask = []
    path_video = []
    target_timings = []
 

    for input, label in batch:
        text = input[0]
        input_list.append(text["input_ids"].tolist()[0])
        text_mask.append(text["attention_mask"].tolist()[0])
        audio_path = input[1]
        vid_features = input[2]  # [6:] for debug
        
        path_video.append(vid_features[0])
        target_timings.append(vid_features[2])
        
        speech_list.append(FEAT(audio_path[1] , sampling_rate=16000).input_features[0])
        video_list.append(vid_features[1])
        

        label_list.append(label)
            
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
        
       
    }

    visual_embeds = {
        "video_path" : path_video,
        "video_embeds": torch.stack(video_list).permute(0, 2, 1, 3, 4),
        
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
            f"{dataset_name}_bert/ylhfsh3s/fresh-sweep-16/best.pt",
            
            f"{dataset_name}_whisper/xaz13i9h/proud-sweep-34/best.pt",
            
            f"{dataset_name}_video/myt3iav2/rose-sweep-69/best.pt",
            ]
        else:
            dataset_name = "must"
            self.must = True
            path = [
            f"{dataset_name}_bert/aoago6hk/flowing-sweep-99/best.pt",
            
            f"{dataset_name}_whisper/qzyfm99j/astral-sweep-24/best.pt",
            
            f"{dataset_name}_video/apl9ufpx/crisp-sweep-51/best.pt",
            ]
            
        print(path , flush = True)
            
        self.f = h5py.File(f"../../data/iemo.TAE.features.hdf5", "r+", libver="latest", swmr=True)
        try:
            self.f.swmr_mode = True
        except:
            pass

        self.p = 0.75

        # self.bert = BertClassifier(args)
        # self.whisper = WhisperForEmotionClassification(args)
        # self.videomae = VideoClassification(args)
        
        

        # Load in our finetuned models, exactly as they should be
        
        # for i, model in enumerate([ self.bert , self.whisper, self.videomae]):
        #     print(f"Loading from {path[i]}, " , flush = True)                           # GO BACK TO CUDA
        #     ckpt =  torch.load(f"../../../TAV_Train/{path[i]}" , map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))['model_state_dict']
        #     model.load_state_dict(ckpt)
        
            
        # ckpt = torch.load("../../results/IEMOCAP/roberta-large/final/2021-05-09-12-19-54-speaker_mode-upper-num_past_utterances-1000-num_future_utterances-0-batch_size-4-seed-4/checkpoint-5975/pytorch_model.bin",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.bert = AutoModel.from_pretrained("roberta-large")
        # new_state_dict = OrderedDict()
        # for k, v in ckpt.items():
        #     name = k.replace("roberta." , "")  # remove 'roberta.' prefix
            
        #     new_state_dict[name] = v

        # # Load the new state dictionary into your model
        # self.bert.load_state_dict(new_state_dict , strict = False)
        # # self.bert.load_state_dict(ckpt)
        # # print(self.bert.classifier.out_proj.bias , flush = True)
        
        
        # # Zero out all gradients. Might not need this anymore though
        # # for i, model in enumerate([self.bert, self.whisper, self.videomae]):
        # for i, model in enumerate([self.bert]):
        #     for param in model.parameters():
        #         param.requires_grad = False

        # # Put the models onto eval mode
        # self.bert.eval()
        
        ckpt = torch.load("../../results/IEMOCAP/roberta-large/final/2021-05-09-12-19-54-speaker_mode-upper-num_past_utterances-1000-num_future_utterances-0-batch_size-4-seed-4/checkpoint-5975/pytorch_model.bin",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.bert2 = AutoModelForSequenceClassification.from_pretrained("roberta-large" , num_labels = 6)
        
        # Load the new state dictionary into your model
        self.bert2.load_state_dict(ckpt , strict = False)
        # self.bert2.load_state_dict(ckpt)
        # print(self.bert2.classifier.out_proj.bias , flush = True)


        # Zero out all gradients. Might not need this anymore though
        # for i, model in enumerate([self.bert2, whisper, videomae]):
        for i, model in enumerate([self.bert2]):
            for param in model.parameters():
                param.requires_grad = False

        # Put the models onto eval mode
        self.bert2.eval()
        
        # self.whisper.eval()
        # self.videomae.eval()

    def forward(
        self,
        input_ids,
        text_attention_mask,
        audio_features,
        
        video_embeds,
        video_mask,
        
        video_path,
        timings,
        
        check="train",
    ):
        # Transformer Time
        # last_hidden_text_state: torch.Tensor
        # text_outputs : torch.Tensor
        # aud_outputs: torch.Tensor
        # aud_context : torch.Tensor
        # vid_outputs: torch.Tensor
        # vid_context : torch.Tensor
        
        
        
        # last_hidden_text_state, text_outputs = self.bert.bert(
        #     input_ids=input_ids, attention_mask=text_attention_mask, return_dict=False
        # )

        outputs = self.bert2(input_ids=input_ids, attention_mask=text_attention_mask)
        logits , last_hidden_text_state = outputs.logits, outputs.hidden_states
        # print("Calculation worked" , flush = True)
        
        # if self.must:
        #     delete = 3
        #     self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/text", data=last_hidden_text_state.cpu().detach().numpy())
        # else:
        #     delete = 3
        # self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text", data=last_hidden_text_state.cpu().detach().numpy())
        self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/new_text", data=last_hidden_text_state.cpu().detach().numpy())
        # data = self.f[f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text"][()]
        # data[...] = last_hidden_text_state.cpu().detach().numpy()
        
        return logits
        
        # print(f"{text_outputs} \n Shape of text_ouptuts is {text_outputs.shape} \n Shape of last_hidden_text_state is {last_hidden_text_state.shape} " , flush = True)

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
                
        #     self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/audio_context", data=new_aud_context.cpu().detach().numpy())
        #     self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/audio", data=aud_outputs.cpu().detach().numpy())
        #     aud_outputs = (aud_outputs.mean(dim=1) * self.p + new_aud_context.mean(dim=1) * (1 - self.p)) / 2
            
        # else:
        # self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/audio", data=aud_outputs.cpu().detach().numpy())
        # aud_outputs = aud_outputs.mean(dim=1)

        # vid_outputs = self.videomae.videomae(video_embeds, bool_masked_pos = video_mask)[0]  

        # if self.must:
        #     vid_context = self.videomae.videomae(video_context, bool_masked_pos = video_mask)[0]
            
        #     self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/video_context", data=vid_context.cpu().detach().numpy())
        #     self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/video", data=vid_outputs.cpu().detach().numpy())
        #     vid_outputs = (vid_outputs[:,0] * self.p + vid_context[:,0] * (1 - self.p)) / 2
        # else:
        # self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/video", data=vid_outputs.cpu().detach().numpy())
        # vid_outputs = vid_outputs[:,0]
        # text_outputs = self.bert.linear1(text_outputs)
        
        # aud_outputs = self.whisper.linear1(aud_outputs)
        
        # vid_outputs = self.videomae.linear1(vid_outputs)
        
        # tav = (text_outputs + aud_outputs + vid_outputs) / 3
        # tav = torch.rand((1 , self.output_dim ),dtype = torch.float32) 
        tav = torch.rand((1 , self.output_dim ),dtype = torch.float32) 
        
        return tav

