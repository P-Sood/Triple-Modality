from glob import glob
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from torch import nn
from transformers import  VideoMAEModel  , AutoModel , AutoProcessor , AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

import numpy as np
from numpy.random import choice
import random
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch , must): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """

    video_list = []
    speech_context = []
    speech_list = []
    label_list = []
    
    if not must:
        video_context = [torch.empty((1,1,1,1)) ,  torch.empty((1,1,1,1))]
    else:
        video_context = []
    speech_list_context_input_values = torch.empty((1))
    speech_list_mask = None
    vid_mask = None


    for (input , label) in batch:
        audio_path = input[0]
        vid_features = input[1]#[6:] for debug
        if not must:
            speech_list.append(audio_path)
            video_list.append(vid_features)
        else:
            speech_list.append(audio_path[0])
            speech_context.append(audio_path[1])
            video_list.append(vid_features[0])
            video_context.append(vid_features[1])
        
        label_list.append(label)
    batch_size = len(label_list)
    
    
    vid_mask = torch.randint(-13, 2, (batch_size, 1568)) # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it, 
    vid_mask[vid_mask > 0] = 0
    vid_mask = vid_mask.bool()
    # now we have a random mask over values so it can generalize better   
    x = torch.count_nonzero(vid_mask.int()).item()
    rem = (1568*batch_size - x)%batch_size
    if rem != 0:
        idx = torch.where(vid_mask.view(-1) == 0)[0]  # get all indicies of 0 in flat tensor
        num_to_change =  rem# as follows from example abow
        idx_to_change = choice(idx, size=num_to_change, replace=False)
        vid_mask.view(-1)[idx_to_change] = 1
    
    speech_list_input_values = pad_sequence(speech_list , batch_first = True, padding_value=0)
    del speech_list
    
    if must:
        speech_list_context_input_values = pad_sequence(speech_context , batch_first = True, padding_value=0)
        del speech_context
    
    
    
    audio_features = {'audio_features':speech_list_input_values , 
            'attention_mask':speech_list_mask, 
            'audio_context':speech_list_context_input_values, 
        }

    visual_embeds = {'visual_embeds':torch.stack(video_list).permute(0,2,1,3,4) , 
            'attention_mask':vid_mask , 
            'visual_context':torch.stack(video_context).permute(0,2,1,3,4) , 
        }
    return [audio_features , visual_embeds] , torch.Tensor(np.array(label_list))


class AudioVideoClassifier(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(AudioVideoClassifier, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']
        self.dataset = args['dataset']
        
        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = .6

        self.test_ctr = 1
        self.train_ctr = 1

        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.wav_2_768_2 = nn.Linear(1024 , 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        self.vid_norm = nn.LayerNorm(768)
        self.aud_norm = nn.LayerNorm(1024)

        
        
        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*2, self.output_dim)
        

    
    def forward(self, audio_features , context_audio , video_embeds , video_context , visual_mask  , check = "train"):        
        aud_outputs = self.wav2vec2(audio_features)[0]
        del audio_features
        aud_outputs = torch.mean(aud_outputs, dim=1)
        aud_outputs = self.aud_norm(aud_outputs)
        if self.must:
            
            aud_context = self.wav2vec2(context_audio)[0]
            del context_audio
            aud_context = torch.mean(aud_context, dim=1)
            aud_context = self.aud_norm(aud_context)
            aud_outputs = (aud_outputs*self.p + aud_context*(1-self.p))/2
            del aud_context
            
        aud_outputs = self.wav_2_768_2(aud_outputs)
        vid_outputs = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        del video_embeds
            
        vid_outputs = torch.mean(vid_outputs, dim=1) # Now it has 2 dimensions 
        vid_outputs = self.vid_norm(vid_outputs) 
        
        if self.must:
            vid_context = self.videomae(video_context , visual_mask)[0]
            del video_context
            vid_context = torch.mean(vid_context, dim=1) # Now it has 2 dimensions 
            vid_context = self.vid_norm(vid_context) 
            vid_outputs = (vid_outputs*self.p + vid_context*(1-self.p))/2
            
        del visual_mask

        #Concatenate Outputs
        tav = torch.cat([ aud_outputs , vid_outputs],dim=1)
        
        del aud_outputs 
        del vid_outputs

        #Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)

        return tav # returns [batch_size,output_dim]