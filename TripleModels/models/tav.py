import numpy as np
import torch
import warnings
from torch import nn
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")



def collate_batch(batch , must):
    text = []
    audio = []
    audio_context = []
    video = []
    video_context = []
    labels = []

    for input, label in batch:
        text.append(input["text_features"].squeeze())
        audio.append(input["audio_features"].squeeze())
        video.append(input["video_features"].squeeze())
        if must:
            audio_context.append(input["audio_context"].squeeze())
            video_context.append(input["video_context"].squeeze())
        labels.append(label)

    text = torch.stack(text, dim=0).squeeze(dim=1)
    audio = torch.stack(audio, dim=0).squeeze(dim=1)
    video = torch.stack(video, dim=0).squeeze(dim=1)
    

    return {
        "text_features": text,
        "audio_features": audio,
        "audio_context": torch.stack(audio_context, dim=0).squeeze(dim=1) if must else torch.Tensor([]),
        "video_features": video,
        "video_context": torch.stack(video_context, dim=0).squeeze(dim=1) if must else torch.Tensor([]),
    }, torch.Tensor(np.array(labels)).long()


class TAVForMAE(nn.Module):
    """
    Model for Multimodal Alignment and Fusion

    Since we have already called an encoder to get all the features for this model, we just need to run the fusion and classifier head over top
    """

    def __init__(self, args):
        super(TAVForMAE, self).__init__()
        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.learn_PosEmbeddings = args["learn_PosEmbeddings"]
        self.num_layers = args["num_layers"]
        self.num_encoders = args["num_encoders"]
        self.dataset = args["dataset"]
        self.fusion = args["fusion"]
        self.hidden_size = args["hidden_size"]

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

        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.75  # This is to decide how much to weight the context vs the actual features for Mustard

        # Everything before this line is unlearnable, everything after is what we are focused on

        self.aud_text_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.vid_text_layers = nn.ModuleList(
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
        elif self.fusion == "sota_dual":
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(1024 * 3, 1024) for _ in range(self.num_layers)]
            )
            self.linear1 = nn.Linear(1024 * 2, self.hidden_size)
        
        elif self.fusion == "guiseppe":
            
            self.linear1 = nn.Linear(1024 * 4, self.hidden_size)
        elif self.fusion == "new_guiseppe":
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(1024 * 3, 1024) for _ in range(self.num_layers)]
            )
            self.linear1 = nn.Linear(1024 * 4, self.hidden_size)
        elif self.fusion == "dual_guiseppe":
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(1024 * 3, 1024) for _ in range(self.num_layers)]
            )
            self.linear1 = nn.Linear(1024 * 4, self.hidden_size)
        elif self.fusion == "concat":
            self.linear1 = nn.Linear(1024 * 3, self.hidden_size)
            

        self.dropout = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.hidden_size, self.output_dim)
        self.relu = nn.ReLU()

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
            
        # Encoder layers
        for i in range(self.num_encoders):
            text_features = self.text_encoder_layers[i](text_features)
            audio_features  = self.audio_encoder_layers[i](audio_features)
            video_features  = self.video_encoder_layers[i](video_features)

        # Model Head
        if self.fusion == "sota":
            
            for i in range(self.num_layers):
                Ffusion1 = text_features
                Ffusion2 = text_features
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                fusion_layer = self.fusion_layers[i]
                # Q, K , V inputs
                # Fuse audio/video to the textual dimension
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, Ffusion1
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, video_features, Ffusion2
                )
                # run a linear layer over audio_text, video_text and text to become the new text features
                text_features = fusion_layer(torch.cat([Ffusion1, Ffusion2 , text_features], dim=-1))
            # Concatenate the text features interlaced with audio and video context, with the audio and video features
            tav = torch.cat([text_features, audio_features, video_features], dim=-1)
        elif self.fusion == "sota_dual":
            
            for i in range(self.num_layers):
                Ffusion1 = text_features
                Ffusion2 = text_features
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                fusion_layer = self.fusion_layers[i]
                # Q, K , V inputs
                # Fuse audio/video to the textual dimension
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, Ffusion1
                )
                Ffusion2, _ = vid_text_layer(
                    audio_features, Ffusion2, audio_features
                )
                # run a linear layer over audio_text, video_text and text to become the new text features
                text_features = fusion_layer(torch.cat([Ffusion1, Ffusion2 , text_features], dim=-1))
            # Concatenate the text features interlaced with audio and video context, with the audio and video features
            tav = torch.cat([text_features, audio_features], dim=-1)
            
        elif self.fusion == "guiseppe":
            # Dont need the fixed MHA encoder here because QV, only need to be the same size
            Ffusion1 = text_features
            Ffusion2 = text_features
            for i in range(self.num_layers):
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                # Query the same, Key and Value are the other modality
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, audio_features
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, video_features, video_features
                )
                # What about updating Ffusion1/2 with a  linear layer like above?
            tav = torch.cat([Ffusion1, Ffusion2, audio_features, video_features], dim=-1)
        elif self.fusion == "new_guiseppe":
            # Dont need the fixed MHA encoder here because QV, only need to be the same size
            for i in range(self.num_layers):
                Ffusion1 = text_features
                Ffusion2 = text_features
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                fusion_layer = self.fusion_layers[i]
                # Query the same, Key and Value are the other modality
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, audio_features
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, video_features, video_features
                )
                text_features = fusion_layer(torch.cat([Ffusion1, Ffusion2 , text_features], dim=-1))
                # What about updating Ffusion1/2 with a  linear layer like above?
            tav = torch.cat([Ffusion1, Ffusion2, audio_features, video_features], dim=-1)
        elif self.fusion == "dual_guiseppe":
            Ffusion1 = text_features
            Ffusion2 = text_features
            for i in range(self.num_layers):
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                # Query the same, Key and Value are the other modality
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, audio_features
                )
                Ffusion2, _ = vid_text_layer(
                    audio_features, Ffusion2, Ffusion2
                )
                # What about updating Ffusion1/2 with a  linear layer like above?
            tav = torch.cat([Ffusion1, Ffusion2 , text_features, audio_features], dim=-1)
        elif self.fusion == "concat":
            tav = torch.cat([text_features, audio_features, video_features], dim=-1)
            
        # batch_size , 512 , 1024*3

        tav = tav.mean(dim=1)  

        del text_features
        del audio_features
        del video_features
        # del Ffusion1
        # del Ffusion2

        # Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)
        tav = self.relu(tav)
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear2(tav)

        return tav  