import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import sys
import random

from transformers import AutoProcessor , AutoModel
from utils.global_functions import pool

from torch.nn.utils.rnn import pad_sequence


# TODO: DATASET SPECIFIC
# PROC = AutoProcessor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

def collate_batch(batch , must): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    speech_list = []
    speech_list_context = []
    label_list = []
    speech_list_mask = None
    speech_list_context_input_values = torch.empty((1))


    for (input , label) in batch:
        if not must:
            speech_list.append(input)
        else:
            speech_list.append(input[0])
            speech_list_context.append(input[1])
            
        # TODO: DATASET SPECIFIC
        label_list.append(label)

    speech_list_input_values = pad_sequence(speech_list , batch_first = True, padding_value=0)
    if must:
        speech_list_context_input_values = pad_sequence(speech_list_context , batch_first = True, padding_value=0)
        
    del speech_list
    
    # speech_list_input_values = (speech_list_input_values1 + speech_list_input_values2)/2
    audio_features = {'audio_features':speech_list_input_values , 
            'attention_mask':speech_list_mask, 
            'audio_context': speech_list_context_input_values,
        }

    return audio_features , torch.Tensor(np.array(label_list))


class Wav2Vec2ForSpeechClassification(nn.Module):
    def __init__(self, args):
        super(Wav2Vec2ForSpeechClassification, self).__init__()
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
        
        self.aud_norm = nn.LayerNorm(1024)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

   
    def forward(self, audio_features , context_audio , check):
        aud_outputs = self.wav2vec2(audio_features)[0]
        aud_outputs = torch.mean(aud_outputs, dim=1)
        aud_outputs = self.aud_norm(aud_outputs)
        if self.must:
            
            aud_context = self.wav2vec2(context_audio)[0]
            aud_context = torch.mean(aud_context, dim=1)
            aud_context = self.aud_norm(aud_context)
            aud_outputs = (aud_outputs*self.p + aud_context*(1-self.p)).mean()
            
        
        if check == "train":
            aud_outputs = self.dropout(aud_outputs)
        aud_outputs = self.linear1(aud_outputs)

        return aud_outputs # returns [batch_size,output_dim]