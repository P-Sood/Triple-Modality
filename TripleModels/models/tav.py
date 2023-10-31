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

class TAVForMAE(nn.Module):
    """
    Model for Multimodal Alignment and Fusion
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
        self.p = 0.6

        # Everything before this line is unlearnable, everything after is what we are focused on

        self.aud_norm = nn.LayerNorm(1024)
        self.bert_norm = nn.LayerNorm(768)
        self.vid_norm = nn.LayerNorm(768)
        
        self.wav_2_768_2 = nn.Linear(1024, 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)

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
        if self.sota:
            self.fusion_layers = nn.ModuleList(
                [
                    nn.Linear(768*2, 768)
                    for _ in range(self.num_layers)
                ]
            )
            self.linear1 = nn.Linear(768 * 3, 768 * 2)
        else:
            self.linear1 = nn.Linear(768 * 4, 768 * 2)
            

        self.dropout = nn.Dropout(self.dropout)            
        self.linear2 = nn.Linear(768 * 2, self.output_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        text_features,
        audio_features,
        audio_context,
        video_features,
        video_context,
        check="train",
    ):
        # Transformer Time
        text_outputs = self.bert_norm(text_features)
        aud_outputs = self.aud_norm(audio_features)
        del text_features
        del audio_features
        
        if self.must:
            audio_context = self.aud_norm(audio_context)
            aud_outputs = (aud_outputs * self.p + audio_context * (1 - self.p)) / 2
            del audio_context

        aud_outputs = self.wav_2_768_2(aud_outputs)

        vid_outputs = self.vid_norm(video_features)
        del video_features

        if self.must:
            video_context = self.vid_norm(video_context)
            vid_outputs = (vid_outputs * self.p + video_context * (1 - self.p)) / 2
            del video_context
        # Model Head
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
        del Ffusion1
        del Ffusion2

        # Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)
        tav = self.relu(tav)
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear2(tav)

        return tav  # returns [batch_size,output_dim]

