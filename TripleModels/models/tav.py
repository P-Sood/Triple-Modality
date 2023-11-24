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

    for input, label in batch:
        text.append(input["text_features"].squeeze())
        audio.append(input["audio_features"].squeeze())
        # audio_context.append(input["audio_context"].squeeze())
        video.append(input["video_features"].squeeze())
        # video_context.append(input["video_context"].squeeze())
        labels.append(label)

    text = torch.stack(text, dim=0).squeeze(dim=1)
    audio = torch.stack(audio, dim=0).squeeze(dim=1)
    # audio_context = pad_sequence(audio_context, batch_first=True)
    video = torch.stack(video, dim=0).squeeze(dim=1)
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
        self.num_encoders = args["num_encoders"]
        self.dataset = args["dataset"]
        self.fusion = args["fusion"]
        self.hidden_size = args["hidden_size"]

        print(f"Using {self.num_layers} layers \nUsing fusion : {self.fusion}", flush=True)
            

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        audio_context: torch.Tensor,
        video_features: torch.Tensor,
        video_context: torch.Tensor,
        check="train",
    ):

        tav = text_features[:,0]
        # Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)

        return tav  
