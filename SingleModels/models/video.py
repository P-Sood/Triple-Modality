import torch
from torch import nn
import numpy as np
from transformers import VideoMAEModel
from numpy.random import choice
import pdb

def collate_batch(batch, must):  # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    label_list = []
    if not must:
        video_context = [torch.empty((1, 1, 1, 1)), torch.empty((1, 1, 1, 1))]
    else:
        video_context = []
    bbox_list = []
    bbox_context = []

    for input, label in batch:
        if not must:
            video_list.append(input[1][0])
            bbox_list.append(input[1][1])
        else:
            video_list.append(input[1][0])
            bbox_list.append(input[1][1])
            video_context.append(input[2][0])
            bbox_context.append(input[2][1])
        label_list.append(label)
        
    batch_size = len(label_list)
    # 14*14*8
    vid_mask = torch.zeros(batch_size, 1568).bool()
    idx = torch.arange(1568)
    idx_to_change = torch.Tensor(choice(idx, size=1056, replace=False))
    # Repeat this tensor batch_size times
    idx_to_change = idx_to_change.repeat(batch_size, 1).long()
    vid_mask[:, idx_to_change] = True

    visual_embeds = {
        "video_embeds": torch.stack(video_list).permute(0, 2, 1, 3, 4),
        "video_context": torch.stack(video_context).permute(0, 2, 1, 3, 4),
        "video_mask": vid_mask,
    }

    return visual_embeds, torch.Tensor(np.array(label_list)).long()


class VideoClassification(nn.Module):
    """A simple ConvNet for binary classification."""

    def __init__(self, args):
        super(VideoClassification, self).__init__()

        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.dataset = args["dataset"]

        self.must = True if "must" in str(self.dataset).lower() or "urfunny" in str(self.dataset).lower() else False
        self.p = 0.75
        self.videomae = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-large"
        )
        
        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

    def forward(self, video_embeds, video_mask, video_context, check="train"):
        # print(video_embeds.shape, flush=True)
        if check == "train":
            vid_outputs = self.videomae(video_embeds, bool_masked_pos = video_mask)[0]
        else:
            vid_outputs = self.videomae(video_embeds)[0]

        vid_outputs = vid_outputs[:, 0]
        # take the first token now it has 2 dimensions
        del video_embeds

        if self.must:
            if check == "train":
                vid_context = self.videomae(video_context, bool_masked_pos = video_mask)[0]
            else:
                vid_context = self.videomae(video_context)[0]
            vid_context = vid_context[:, 0] 
            del video_context
            vid_outputs = (vid_outputs * self.p + vid_context * (1 - self.p)) / 2
            del vid_context

        if check == "train":
            vid_outputs = self.dropout(vid_outputs)
        vid_outputs = self.linear1(vid_outputs)

        return vid_outputs  # returns [batch_size,output_dim]
