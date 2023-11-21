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
        "audio_context": torch.Tensor([]),
        "video_features": video,
        "video_context": torch.Tensor([]),
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
        self.dataset = args["dataset"]
        self.fusion = args["fusion"]
        self.hidden_size = args["hidden_size"]

        print(f"Using {self.num_layers} layers \nUsing fusion : {self.fusion}", flush=True)

        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.6  # This is to decide how much to weight the context vs the actual features for Mustard

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
        if self.fusion:
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(1024 * 3, 1024) for _ in range(self.num_layers)]
            )
            self.linear1 = nn.Linear(1024 * 3, self.hidden_size)
        else:
            self.linear1 = nn.Linear(1024 * 4, self.hidden_size)

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


        text_audio_video_aligned = pad_sequence(
            torch.unbind(text_features, dim=0)
            + torch.unbind(audio_features, dim=0)
            + torch.unbind(video_features, dim=0),
            batch_first=True,
        )
        bs = len(text_audio_video_aligned) // 3
        text_features, audio_features, video_features = (
            text_audio_video_aligned[:bs],
            text_audio_video_aligned[bs : 2 * bs],
            text_audio_video_aligned[2 * bs :],
        )
        # Create the padding mask for the aud tensor
        audio_mask = torch.any(audio_features == 0, dim=-1)

        # Create the padding mask for the video tensor
        video_mask = torch.any(video_features == 0, dim=-1)
        # Pad the sequence length dimension based on batch sizes. This is because the MHA expects a fixed sequence length
        # Model Head
        if self.fusion == "sota":
            for i in range(self.num_layers):
                Ffusion1 = text_features
                Ffusion2 = text_features
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                fusion_layer = self.fusion_layers[i]
                # Q, K , V inputs
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, Ffusion1, key_padding_mask=audio_mask
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, video_features, Ffusion2, key_padding_mask=video_mask
                )
                text_features = fusion_layer(torch.cat([Ffusion1, Ffusion2 , text_features], dim=-1))
            tav = torch.cat([text_features, audio_features, video_features], dim=-1)
        elif self.fusion == "guiseppe":
            # Dont need the fixed MHA encoder here because QV, only need to be the same size
            Ffusion1 = text_features
            Ffusion2 = text_features
            for i in range(self.num_layers):
                aud_text_layer = self.aud_text_layers[i]
                vid_text_layer = self.vid_text_layers[i]
                # Key the same, Query and Value are the other modality
                Ffusion1, _ = aud_text_layer(
                    Ffusion1, audio_features, audio_features, key_padding_mask=audio_mask
                )
                Ffusion2, _ = vid_text_layer(
                    Ffusion2, video_features, video_features, key_padding_mask=video_mask
                )
            tav = torch.cat([Ffusion1, Ffusion2, audio_features, video_features], dim=-1)
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
