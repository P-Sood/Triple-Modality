import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
import h5py
import random
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)


# ------------------------------------------------------------TRIPLE MODELS BELOW--------------------------------------------------------------------
class TextAudioVideoDataset(Dataset):
    """
    feature_col1 : audio paths
    feature_col2 : video paths
    feature_col3 : text
    """

    def __init__(
        self,
        df,
        dataset,
        batch_size,
        feature_col1,
        feature_col2,
        feature_col3,
        label_col,
        timings=None,
        accum=False,
        check="test",
    ):
        self.audio_path = df[feature_col1].values
        self.video_path = df[feature_col2].values
        data_path = "../../data/"
        dataset = str(dataset).lower()

        if timings != None:
            try:
                self.timings = df[timings].values.tolist()
            except:
                self.timings = [None] * len(self.audio_path)
        else:
            self.timings = [None] * len(self.audio_path)

        if "meld" in dataset:
            dataset = "meld"
        elif "iemo" in dataset:
            dataset = "iemo"
        elif "tiktok" in dataset:
            dataset = "tiktok"
        else:
            dataset = "mustard"
            self.timings = df["timings"].values.reshape(-1, 2).tolist()
            self.audio_path = df[feature_col1].values.reshape(-1, 2).tolist()
            self.video_path = df[feature_col2].values.reshape(-1, 2).tolist()
            df = df[df["context"] == False]

        fh = f"{data_path}{dataset}.features.hdf5"

        # if "meld" in str(dataset).lower():
        #     file = "../../data/meld_features.hdf5"
        # elif "iemo" in str(dataset).lower():
        #     file = "../../data/iemo_features.hdf5"
        # elif "tiktok" in str(dataset).lower():
        #     file = "../../data/tiktok_features.hdf5"
        # else:
        #     # Need to reshape the columns because it is context followed by target constantly. Thus only half of the rows are targets.
        #     file = "../../data/must_features.hdf5"
        #     self.timings = df["timings"].values.reshape(-1, 2).tolist()
        #     self.audio_path = df[feature_col1].values.reshape(-1, 2).tolist()
        #     self.video_path = df[feature_col2].values.reshape(-1, 2).tolist()
        #     df = df[df["context"] == False]

        self.Data = Data(file = fh)
        self.check = check
        self.labels = df[label_col].values.tolist()

        assert (
            len(self.audio_path) == len(self.video_path)
        ), "wrong lengths"

        if accum:
            self.grad = (
                (df["dialog"].value_counts() / batch_size)
                .astype(int)
                .sort_index()
                .tolist()
            )
            self.grad_sum = [sum(self.grad[: i + 1]) for i, x in enumerate(self.grad)]
            self.ctr = 0

    def retGradAccum(self, i: int) -> int:
        RETgrad = self.grad[self.ctr]
        RETgrad_sum = self.grad_sum[self.ctr]
        if i + 1 == self.grad_sum[self.ctr]:
            self.ctr += 1
        if self.ctr == len(self.grad):
            self.resetCtr()
        return RETgrad, RETgrad_sum

    def resetCtr(self):
        self.ctr = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.Data.textFeatures(self.video_path[idx], self.timings[idx], self.check)
        audio, audio_context = self.Data.audioFeatures(self.video_path[idx], self.timings[idx], self.check)
        video, video_context = self.Data.videoFeatures(self.video_path[idx], self.timings[idx], self.check)
        return { "text_features": torch.Tensor(text),
        "audio_features" : torch.Tensor(audio),
        "audio_context"  : torch.Tensor(audio_context),
        "video_features" : torch.Tensor(video),
        "video_context"  : torch.Tensor(video_context),

        } , torch.Tensor(np.array(self.labels[idx])).long()


# ------------------------------------------------------------DOUBLE MODELS BELOW--------------------------------------------------------------------
class Data:
    """
    Just get the features from hdf5 and return them to the dataloader for our loops
    This would technically be our collate batch function
    """
    def __init__(self, file) -> None:
        self.FILE = h5py.File(file, "r", libver="latest", swmr=True)
        self.must = False

        self.must = True if "must" in file else False
        self.iemo = True if "iemo" in file else False
        self.tiktok = True if "tiktok" in file else False
        self.meld = True if "meld" in file else False

    def videoFeatures(self, path, timings, check):
        if not self.must:
            video = torch.Tensor(self.FILE[f"{check}/{path.split('/')[-1][:-4]}_{timings}/video"][()])
            return video, None
        else:
            video = torch.Tensor(self.FILE[f"{check}/{path[0].split('/')[-1][:-4]}_{timings[0]}/video"][()])
            video_context = torch.Tensor(self.FILE[f"{check}/{path[1].split('/')[-1][:-4]}_{timings[1]}/video_context"][()])
            return video , video_context
    def audioFeatures(self, path, timings, check):
        if not self.must:
            audio = torch.Tensor(self.FILE[f"{check}/{path[0].split('/')[-1][:-4]}_{timings[0]}/audio"][()])
            return audio , None
        else:
            audio = torch.Tensor(self.FILE[f"{check}/{path.split('/')[-1][:-4]}_{timings}/audio"][()])
            audio_context = torch.Tensor(self.FILE[f"{check}/{path[1].split('/')[-1][:-4]}_{timings[1]}/audio_context"][()])
            return audio , audio_context
    def textFeatures(self, path, timings, check):
        text = torch.Tensor(self.FILE[f"{check}/{path.split('/')[-1][:-4]}_{timings}/text"][()])
        return text
