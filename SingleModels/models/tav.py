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
import gc


FEAT = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

def collate_batch(batch, must):  # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """

    video_list = []
    if not must:
        video_list_context = [torch.empty((1, 1, 1, 1)), torch.empty((1, 1, 1, 1))]
    else:
        video_list_context = []

    speech_list = []
    speech_list_context = []
    label_list = []
    input_list = []
    text_mask = []
    aud_path = []
    path_video = []
    target_timings = []
 

    for input, label in batch:
        text = input[0]
        input_list.append(text["input_ids"].tolist()[0])
        text_mask.append(text["attention_mask"].tolist()[0])
        audio_path = input[1]
        vid_features = input[2]  # [6:] for debug
        vid_bbox = []
        vid_context_bbox = []
        
        if not must:
            path , speech_array , aud_timings = audio_path
            vid_path , video, vid_timings = vid_features
            path_video.append(vid_path)
            target_timings.append(vid_timings)
            
            speech_list.append(FEAT(speech_array , sampling_rate=16000).input_features[0])
            video_list.append(video[0])
            vid_bbox.append(video[1])
            
        else:
            aud_path , speech_array_target, speech_array_context , aud_timings = audio_path 
            vid_path , video_target, video_context, vid_timings = vid_features
            
            path_video.append(vid_path)
            target_timings.append(vid_timings)
            
            speech_list.append(FEAT(speech_array_target , sampling_rate=16000).input_features[0])
            speech_list_context.append(FEAT(speech_array_context , sampling_rate=16000).input_features[0])
            
            video_list.append(video_target[0])
            vid_bbox.append(video_target[1])
            video_list_context.append(video_context[0])
            vid_context_bbox.append(video_context[1])

            del speech_array_target
            del speech_array_context
            del video_target
            del video_context
            gc.collect()
        

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
        "context_audio": torch.Tensor(np.array(speech_list_context)),
        
       
    }

    visual_embeds = {
        "video_path" : path_video,
        "video_embeds": torch.stack(video_list).permute(0, 2, 1, 3, 4),
        "context_video": torch.stack(video_list_context).permute(0, 2, 1, 3, 4),
        
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
        self.dataset = args["dataset"]
        self.early_stop = args["early_stop"]
                
        self.must = False
        if "iemo" in str(self.dataset).lower():
            dataset_name = "iemo"
            path = [
            f"{dataset_name}_bert/ylhfsh3s/fresh-sweep-16/best.pt",
            
            f"Iemo_Whisper_F1_Final/e5g03ntz/eager-sweep-14/best.pt",
            
            f"Iemo_Video_F1_Final/fxztze3d/visionary-sweep-26/best.pt",
            ]
        elif "must" in str(self.dataset).lower():
            dataset_name = "must"
            self.must = True
            path = [
            f"Triple-Modality-SingleModels/None/olive-flower-49/best.pt",
            
            f"Must_Whisper_1500/xcwmnly0/eager-sweep-3/best.pt",
            
            f"Must_Video_1568/luqfajb5/young-sweep-8/best.pt",
            ]
        else:
            dataset_name = "UrFunny"
            self.must = True
            path = [
            f"UrFunny_Text_Final_4_Steps/2bq5ig7b/clean-sweep-2/best.pt",
            
            f"UrFunny_Whisper_Final_4_Steps/drojten5/smooth-sweep-5/best.pt",
            
            f"UrFunny_Video_Final_4_Steps/yhf1da0r/ruby-sweep-10/best.pt",
            ]
            
        print(path , flush = True)
            
        self.f = h5py.File(f"../../data/{dataset_name.lower()}.{self.early_stop}.total.seq_len.features.hdf5", "a", libver="latest", swmr=True)
        try:
            self.f.swmr_mode = True
        except:
            pass

        self.p = 0.75

        self.bert = BertClassifier(args)
        self.whisper = WhisperForEmotionClassification(args)
        self.videomae = VideoClassification(args)
        
        

        # Load in our finetuned models, exactly as they should be
        
        for i, model in enumerate([self.bert, self.whisper, self.videomae] if self.must else [self.whisper, self.videomae]):
            print(f"Loading from {path[i]}, " , flush = True)                           # GO BACK TO CUDA
            ckpt =  torch.load(f"../../../TAV_Train/{path[i]}" , map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))['model_state_dict']
            model.load_state_dict(ckpt)        
        
        if not self.must:
            ckpt = torch.load("../../results/IEMOCAP/roberta-large/final/2021-05-09-12-19-54-speaker_mode-upper-num_past_utterances-1000-num_future_utterances-0-batch_size-4-seed-4/checkpoint-5975/pytorch_model.bin",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.bert = AutoModelForSequenceClassification.from_pretrained("roberta-large" , num_labels = 6)
            self.bert.load_state_dict(ckpt , strict = False)

        
        # Zero out all gradients. Might not need this anymore though
        for i, model in enumerate([self.bert, self.whisper, self.videomae]):
            for param in model.parameters():
                param.requires_grad = False

        # Put the models onto eval mode
        self.bert.eval()
        self.whisper.eval()
        self.videomae.eval()
        

    def forward(
        self,
        input_ids,
        text_attention_mask,
        audio_features,
        context_audio,
        
        video_embeds,
        context_video,
        video_mask,
        
        video_path,
        timings,
        
        check="train",
    ):
        if self.must:
            last_hidden_text_state, text_outputs = self.bert.bert(
                input_ids=input_ids, attention_mask=text_attention_mask, return_dict=False
            )
        else:
            # IEMO ONLY
            outputs = self.bert(input_ids=input_ids, attention_mask=text_attention_mask)
            text_outputs , last_hidden_text_state = outputs.logits, outputs.hidden_states
            
        del input_ids
        del text_attention_mask
        if self.must:
            delete = 3
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/text", data=last_hidden_text_state.cpu().detach().numpy())
        else:
            delete = 3
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/text", data=last_hidden_text_state.cpu().detach().numpy())
        
        aud_outputs = self.whisper.whisper.encoder(audio_features)[0]
        del audio_features
        if self.must:
            aud_context = self.whisper.whisper.encoder(context_audio)[0]
            del context_audio
                
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/audio_context", data=aud_context.cpu().detach().numpy())
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/audio", data=aud_outputs.cpu().detach().numpy())
            aud_outputs = (aud_outputs.mean(dim=1) * self.p + aud_context.mean(dim=1) * (1 - self.p)) / 2
            
        else:
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/audio", data=aud_outputs.cpu().detach().numpy())
            aud_outputs = aud_outputs.mean(dim=1)

        vid_outputs = self.videomae.videomae(video_embeds)[0]  
        del video_embeds
        if self.must:
            vid_context = self.videomae.videomae(context_video)[0]
            del context_video
            del video_mask
            self.f.create_dataset(f"{check}/{video_path[0][0].split('/')[-1][:-4]}_{timings[0][0]}/video_context", data=vid_context.cpu().detach().numpy())
            self.f.create_dataset(f"{check}/{video_path[0][1].split('/')[-1][:-4]}_{timings[0][1]}/video", data=vid_outputs.cpu().detach().numpy())
            vid_outputs = (vid_outputs[:,0] * self.p + vid_context[:,0] * (1 - self.p)) / 2
        else:
            del video_mask
            self.f.create_dataset(f"{check}/{video_path[0].split('/')[-1][:-4]}_{timings[0]}/video", data=vid_outputs.cpu().detach().numpy())
            vid_outputs = vid_outputs[:,0]
        del timings
        del video_path
        
        # text_outputs = self.bert.linear1(text_outputs)
        aud_outputs = self.whisper.linear1(aud_outputs)
        vid_outputs = self.videomae.linear1(vid_outputs)
        return vid_outputs
        # return vid_outputs
        tav = (text_outputs + aud_outputs + vid_outputs) / 3
        
        return tav