import numpy as np
import torch
import warnings
from torch import nn
from transformers import logging
from torch.nn.utils.rnn import pad_sequence

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
import pdb

def collate_batch(batch , must):
    text = []
    audio = []
    audio_context = []
    video = []
    video_context = []
    labels = []

    for input, label in batch:
        itf = input["text_features"]
        atf = input["audio_features"]
        vtf = input["video_features"]
        text.append(itf.squeeze()  if len(itf.shape) > 2 else itf)
        audio.append(atf.squeeze() if len(atf.shape) > 2 else atf)
        video.append(vtf.squeeze() if len(vtf.shape) > 2 else vtf)
        if must:
            audio_context.append(input["audio_context"].squeeze())
            video_context.append(input["video_context"].squeeze())
        labels.append(label)
    try:
        text = pad_sequence(text  , batch_first=True).squeeze(dim=1)
        audio = pad_sequence(audio, batch_first=True).squeeze(dim=1)
        video = pad_sequence(video, batch_first=True).squeeze(dim=1)
    except:
        breakpoint()
    

    return {
        "text_features": text,
        "audio_features": audio,
        "audio_context": torch.stack(audio_context, dim=0).squeeze(dim=1) if must else torch.Tensor([]),
        "video_features": video,
        "video_context": torch.stack(video_context, dim=0).squeeze(dim=1) if must else torch.Tensor([]),
    }, torch.Tensor(np.array(labels)).long()
    


import pdb
class TAVForMAE(nn.Module):
    """
    Model for Multimodal Alignment and Fusion

    Since we have already called an encoder to get all the features for this model, we just need to run the fusion and classifier head over top
    """

    def __init__(self, args):
        super(TAVForMAE, self).__init__()
        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.num_layers = args["num_layers"]
        self.num_encoders = args["num_encoders"]
        self.dataset = args["dataset"]
        self.fusion = args["fusion"]
        self.hidden_size = args["hidden_size"]
        self.must = True if "must" in str(self.dataset).lower() or "urfunny" in str(self.dataset).lower() else False
        self.mosei = True if "mosei" in str(self.dataset).lower() else False
        self.p = 0.75  # This is to decide how much to weight the context vs the actual features for Mustard

        print(f"Using {self.num_layers} layers \nUsing fusion : {self.fusion}", flush=True)
                    
        self.text_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=1024, nhead=8, dropout=self.dropout , batch_first=True)
                for _ in range(self.num_encoders)
            ]
        )
        self.audio_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=1024, nhead=8, dropout=self.dropout , batch_first=True)
                for _ in range(self.num_encoders)
            ]
        )
        self.video_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=1024, nhead=8, dropout=self.dropout , batch_first=True)
                for _ in range(self.num_encoders)
            ]
        )


        self.layers1 = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.layers2 = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        
        
            
            
        if self.fusion == "sota":
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(1024 * 3, 1024) for _ in range(self.num_layers)]
            )
            self.linear1 = nn.Linear(1024 * 3, self.hidden_size)
        
        elif "t_p" in self.fusion:
            self.layers3 = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                    for _ in range(self.num_layers)
                ]
            )
            # self.layers4 = nn.ModuleList(
            #     [
            #         nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
            #         for _ in range(self.num_layers)
            #     ]
            # )
            
            self.linear1 = nn.Linear(1024 * 3 , self.hidden_size)
            
        elif "dp" in self.fusion:
            self.linear1 = nn.Linear(1024 * 4, self.hidden_size)
        elif "d_c" in self.fusion: # double concat
            self.linear1 = nn.Linear(1024 * 2, self.hidden_size)
        elif self.fusion == "t_c": # triple concat
            self.linear1 = nn.Linear(1024 * 3, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.hidden_size, self.output_dim)
        self.relu = nn.ReLU()
    
    def dual_peppe(self, feature1 , feature2, fusion_layer1 , fusion_layer2):
            Ffusion1 = feature1 # text
            Ffusion2 = feature1 # text
            for i in range(self.num_layers):
                layer1 = fusion_layer1[i]
                layer2 = fusion_layer2[i]
                # Query the same, Key and Value are the other modality
                Ffusion1, _ = layer1(
                    Ffusion1, feature2, feature2
                )
                # first fusion we modify the query at every step
                Ffusion2, _ = layer2(
                    feature2, Ffusion2, Ffusion2
                )
                # second fusion we modify the key,value at every step

            dual = torch.cat([Ffusion1.mean(dim=1), Ffusion2.mean(dim=1), 
                             feature1.mean(dim=1), feature2.mean(dim=1)], 
                             dim=-1)
            del Ffusion1
            del Ffusion2
            return dual # Now it has only 2 dimensions

    def triple_peppe(self, text , audio, video):
            Ffusion1 = text # text
            Ffusion2 = text # text
            Ffusion3 = text # text
            for i in range(self.num_layers):
                layer1 = self.layers1[i]
                layer2 = self.layers2[i]
                layer3 = self.layers3[i]

                #1 keep text as query
                #2 relevant modality as query, and use text as our

                # Query the same, Key and Value are the other modality
                Ffusion1, _ = layer1(
                    Ffusion1, audio, audio
                )
                # first fusion we modify the query at every step
                Ffusion2, _ = layer2(
                    audio, Ffusion2, Ffusion2
                )

                Ffusion3, _ = layer3(
                    video, Ffusion3, Ffusion3
                )
                # second fusion we modify the key,value at every step

            triple = torch.cat(
                [
                    Ffusion1.mean(dim=1), Ffusion2.mean(dim=1), Ffusion3.mean(dim=1)
                ], 
            dim=-1)
            del Ffusion1
            del Ffusion2
            del Ffusion3
            return triple # Now it has only 2 dimensions
        

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        audio_context: torch.Tensor,
        video_features: torch.Tensor,
        video_context: torch.Tensor,
        check="train",
    ):
        # Transformer Time
        if self.must:
            audio_features = (audio_features * self.p + audio_context * (1 - self.p)) / 2
            del audio_context
            video_features = (video_features * self.p + video_context * (1 - self.p)) / 2
            del video_context
        
        if self.mosei:
            text_features, _ = self.text_expansion(text_features.float())
            audio_features, _ = self.audio_expansion(audio_features.float())
            video_features, _ = self.video_expansion(video_features.float())
            
        # Encoder layers
        for i in range(self.num_encoders):
            text_features = self.text_encoder_layers[i](text_features)
            audio_features  = self.audio_encoder_layers[i](audio_features)
            video_features  = self.video_encoder_layers[i](video_features)

        # Model Head
                    
        if self.fusion == "d_c":
            tav = torch.cat([text_features, audio_features], dim=-1).mean(dim=1)  
        elif self.fusion == "d_c_av":
            tav = torch.cat([video_features, audio_features], dim=-1).mean(dim=1)  
        elif self.fusion == "d_c_tv":
            tav = torch.cat([text_features, video_features], dim=-1).mean(dim=1)  
        elif self.fusion == "t_c":
            tav = torch.cat([text_features, audio_features, video_features], dim=-1).mean(dim=1)  
        elif self.fusion == "dp_tv":
            tav = self.dual_peppe(text_features , video_features, self.layers1 , self.layers2)
        elif self.fusion == "dp_av":
            tav = self.dual_peppe(audio_features , video_features, self.layers1 , self.layers2)
        elif self.fusion == "dp_ta":
            tav = self.dual_peppe(text_features , audio_features, self.layers1 , self.layers2)
        elif self.fusion == "t_p":
            tav = self.triple_peppe(text_features , audio_features, video_features)
        

        del text_features
        del audio_features
        del video_features

        # Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)
        tav = self.relu(tav)
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear2(tav)

        return tav  
    # python3 ../tav_nn.py --fusion t_p --dataset ../../data/iemo --label_task emotion --sampler Both_NoAccum



"""
    So get the outputs of the text attended wiht audio, and text attended with video. 
    combine and concatenate and that becomes our new "text" which we then supply back to audio/video
"""
