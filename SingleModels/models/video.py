import torch
from torch import nn
import numpy as np
from transformers import VideoMAEModel
from numpy.random import choice


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

    for input, label in batch:
        if not must:
            video_list.append(input[1])
        else:
            video_list.append(input[1])
            video_context.append(input[2])
        label_list.append(label)

    batch_size = len(label_list)
    vid_mask = torch.randint(
        -13, 2, (batch_size, 1568)
    )  # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it,
    vid_mask[vid_mask > 0] = 0
    vid_mask = vid_mask.bool()
    # now we have a random mask over values so it can generalize better
    x = torch.count_nonzero(vid_mask.int()).item()
    rem = (1568 * batch_size - x) % batch_size
    if rem != 0:
        idx = torch.where(vid_mask.view(-1) == 0)[
            0
        ]  # get all indicies of 0 in flat tensor
        num_to_change = rem  # as follows from example abow
        idx_to_change = choice(idx, size=num_to_change, replace=False)
        vid_mask.view(-1)[idx_to_change] = 1

    visual_embeds = {
        "video_embeds": torch.stack(video_list).permute(0, 2, 1, 3, 4),
        "visual_mask": vid_mask,
        "video_context": torch.stack(video_context).permute(0, 2, 1, 3, 4),
    }

    return visual_embeds, torch.Tensor(np.array(label_list))


class VideoClassification(nn.Module):
    """A simple ConvNet for binary classification."""

    def __init__(self, args):
        super(VideoClassification, self).__init__()

        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.learn_PosEmbeddings = args["learn_PosEmbeddings"]
        self.num_layers = args["num_layers"]
        self.dataset = args["dataset"]

        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.6
        self.videomae = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-large"
        )
        
        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

    def forward(self, video_embeds, video_context, visual_mask, check="train"):
        vid_outputs = self.videomae(video_embeds, visual_mask)[0][:, 0] 
        vid_outputs = vid_outputs[:, 0] 
        # take the first token now it has 2 dimensions
        del video_embeds
    

        if self.must:
            vid_context = self.videomae(video_context, visual_mask)[0]
            vid_context = vid_context[:, 0] 
            del video_context
            vid_outputs = (vid_outputs * self.p + vid_context * (1 - self.p)) / 2

        del visual_mask

        if check == "train":
            vid_outputs = self.dropout(vid_outputs)
        vid_outputs = self.linear1(vid_outputs)

        return vid_outputs  # returns [batch_size,output_dim]
