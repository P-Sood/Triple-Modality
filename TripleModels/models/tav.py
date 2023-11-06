import numpy as np
import torch
import warnings
from torch import nn
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    text = []
    audio = []
    audio_context = []
    video = []
    video_context = []
    labels = []

    # text = batch[0]["text_features"]#.squeeze()
    # audio = batch[0]["audio_features"]#.squeeze()
    # audio_context = batch[0]["audio_context"]#.squeeze()
    # video = batch[0]["video_features"]#.squeeze()
    # video_context = batch[0]["video_context"]#.squeeze()

    for input, label in batch:
        text.append(input["text_features"].squeeze())
        audio.append(input["audio_features"].squeeze())
        # audio_context.append(input["audio_context"].squeeze())
        video.append(input["video_features"].squeeze())
        # video_context.append(input["video_context"].squeeze())
        labels.append(label)

    text = torch.stack(text, dim=0).squeeze(dim=1)
    audio = pad_sequence(audio, batch_first=True).squeeze(dim=1)
    # audio_context = pad_sequence(audio_context, batch_first=True)
    video = pad_sequence(video, batch_first=True).squeeze(dim=1)
    # video_context = pad_sequence(video_context, batch_first=True)

    return {
        "text_features": text,
        "audio_features": audio,
        # "audio_context"  : audio_context,
        "audio_context": torch.Tensor([]),
        "video_features": video,
        "video_context": torch.Tensor([]),
        # "video_context"  : video_context,
    }, torch.Tensor(np.array(labels)).long()


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
        self.num_encoders = args["num_encoders"]
        self.dataset = args["dataset"]
        self.sota = args["sota"]
        self.hidden_size = args["hidden_size"]

        print(f"Using {self.num_layers} layers \nUsing sota = {self.sota}", flush=True)

        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.6  # This is to decide how much to weight the context vs the actual features for Mustard

        # Everything before this line is unlearnable, everything after is what we are focused on

        self.aud_norm = nn.LayerNorm(1024)
        self.bert_norm = nn.LayerNorm(768)
        self.vid_norm = nn.LayerNorm(768)
        self.wav_2_768_2 = nn.Linear(1024, 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)

        self.aud_text_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.vid_text_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        
        self.text_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
                for _ in range(self.num_encoders)
            ]
        )
        self.audio_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
                for _ in range(self.num_encoders)
            ]
        )
        self.video_encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
                for _ in range(self.num_encoders)
            ]
        )
        
        if self.sota:
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(768 * 2, 768) for _ in range(self.num_layers)]
            )
            self.linear1 = nn.Linear(768 * 3, self.hidden_size)
        else:
            self.linear1 = nn.Linear(768 * 4, self.hidden_size)

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
            
        text_mask = torch.any(text_outputs == 0, dim=-1)
        audio_mask = torch.any(aud_outputs == 0, dim=-1)
        video_mask = torch.any(vid_outputs == 0, dim=-1)    
        for i in range(self.num_encoders):
            text_outputs = self.text_encoder_layers[i](text_outputs , src_key_padding_mask = text_mask)
            aud_outputs  = self.audio_encoder_layers[i](aud_outputs , src_key_padding_mask = audio_mask)
            vid_outputs  = self.video_encoder_layers[i](vid_outputs , src_key_padding_mask = video_mask)

        text_audi_video_aligned = pad_sequence(
            torch.unbind(text_outputs, dim=0)
            + torch.unbind(aud_outputs, dim=0)
            + torch.unbind(vid_outputs, dim=0),
            batch_first=True,
        )
        bs = len(text_audi_video_aligned) // 3
        text_outputs, aud_outputs, vid_outputs = (
            text_audi_video_aligned[:bs],
            text_audi_video_aligned[bs : 2 * bs],
            text_audi_video_aligned[2 * bs :],
        )
        del text_audi_video_aligned
        # Create the padding mask for the aud tensor
        audio_mask = torch.any(aud_outputs == 0, dim=-1)

        # Create the padding mask for the video tensor
        video_mask = torch.any(vid_outputs == 0, dim=-1)
        # Pad the sequence length dimension based on batch sizes. This is because the MHA expects a fixed sequence length
        # Model Head
        if self.sota:
            for i in range(self.num_layers):
                Ffusion1 = text_outputs
                Ffusion2 = text_outputs
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                fusion_layer = self.fusion_layers[i]
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, aud_outputs, Ffusion1, key_padding_mask=audio_mask
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, vid_outputs, Ffusion2, key_padding_mask=video_mask
                )
                text_outputs = fusion_layer(torch.cat([Ffusion1, Ffusion2], dim=-1))
            tav = torch.cat([text_outputs, aud_outputs, vid_outputs], dim=-1)
        else:
            # Dont need the fixed MHA encoder here because QV, only need to be the same size
            Ffusion1 = text_outputs
            Ffusion2 = text_outputs
            for i in range(self.num_layers):
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, aud_outputs, aud_outputs, key_padding_mask=audio_mask
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, vid_outputs, vid_outputs, key_padding_mask=video_mask
                )
            tav = torch.cat([Ffusion1, Ffusion2, aud_outputs, vid_outputs], dim=-1)

        tav = tav.mean(dim=1)  # I take the mean here

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
