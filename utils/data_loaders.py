import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pdb


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
                df['timings'] = df['timings'].replace({np.nan:None})
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
            dataset = "must"
            self.timings = df["timings"].values.reshape(-1, 2).tolist()
            self.audio_path = df[feature_col1].values.reshape(-1, 2).tolist()
            self.video_path = df[feature_col2].values.reshape(-1, 2).tolist()
            df = df[df["context"] == False]

        # fh = f"{data_path}{dataset}.final.seq_len.features.hdf5"
        fh = "/home/zeerak.talat/trimodal/data/iemo.TAE.features.hdf5"

        self.Data = Data(file=fh)
        self.check = check
        self.labels = df[label_col].values.tolist()

        assert len(self.audio_path) == len(self.video_path), "wrong lengths"

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
        timings = self.timings[idx]
        # timings = (
        #     list(self.timings[idx]) if (self.timings[idx] != None or self.timings[idx] != np.nan) else self.timings[idx]
        # )
        timings = self.timings[idx]
        text = self.Data.textFeatures(self.video_path[idx], timings, self.check)
        audio, audio_context = self.Data.audioFeatures(
            self.video_path[idx], timings, self.check
        )
        video, video_context = self.Data.videoFeatures(
            self.video_path[idx], timings, self.check
        )

        return {
            "text_features": text,
            "audio_features": audio,
            "audio_context": audio_context,
            "video_features": video,
            "video_context": video_context,
        }, torch.tensor(np.array(self.labels[idx])).long()


# ------------------------------------------------------------DOUBLE MODELS BELOW--------------------------------------------------------------------
class Data:
    """
    Just get the features from hdf5 and return them to the dataloader for our loops
    This would technically be our collate batch function
    """

    def __init__(self, file) -> None:
        self.FILE = h5py.File(file, "r", libver="latest", swmr=True)
        self.must = True if "must" in file else False
        self.iemo = True if "iemo" in file else False
        self.tiktok = True if "tiktok" in file else False
        self.meld = True if "meld" in file else False

    def videoFeatures(self, path, timings, check):
        if not self.must:
            video = torch.tensor(
                self.FILE[f"{check}/{path.split('/')[-1][:-4]}_{timings}/video"][()]
            )
            return video, torch.Tensor([])
        else:
            
            try:
                video_context = torch.tensor(
                    self.FILE[f"{check}/{path[0].split('/')[-1][:-4]}_{list(timings[0])}/video_context"][
                        ()
                    ]
                )
            except:
                video_context = torch.tensor(
                    self.FILE[f"{check}/{path[0].split('/')[-1][:-4]}_{timings[0]}/video_context"][
                        ()
                    ]
                )
            try:
                video = torch.tensor(
                    self.FILE[
                        f"{check}/{path[1].split('/')[-1][:-4]}_{list(timings[1])}/video"
                    ][()]
                )
            except:
                video = torch.tensor(
                    self.FILE[
                        f"{check}/{path[1].split('/')[-1][:-4]}_{timings[1]}/video"
                    ][()]
                )
            return video, video_context

    def audioFeatures(self, path, timings, check):
        if not self.must:
            audio = torch.tensor(
                self.FILE[f"{check}/{path.split('/')[-1][:-4]}_{timings}/audio"][()]
            )
            return audio, torch.Tensor([])
        else:
            try:
                audio_context = torch.tensor(
                    self.FILE[f"{check}/{path[0].split('/')[-1][:-4]}_{list(timings[0])}/audio_context"][
                        ()
                    ]
                )
            except:
                audio_context = torch.tensor(
                    self.FILE[f"{check}/{path[0].split('/')[-1][:-4]}_{timings[0]}/audio_context"][
                        ()
                    ]
                )
            try:
                audio = torch.tensor(
                    self.FILE[
                        f"{check}/{path[1].split('/')[-1][:-4]}_{list(timings[1])}/audio"
                    ][()]
                )
            except:
                audio = torch.tensor(
                    self.FILE[
                        f"{check}/{path[1].split('/')[-1][:-4]}_{timings[1]}/audio"
                    ][()]
                )
            return audio, audio_context

    def textFeatures(self, path, timings, check):

        if not self.must:
            text = torch.tensor(
                self.FILE[f"{check}/{path.split('/')[-1][:-4]}_{timings}/text"][()]
            )
                
        else:
            try:
                text = torch.tensor(
                    self.FILE[f"{check}/{path[1].split('/')[-1][:-4]}_{timings[1]}/text"][()]
                )
            except:
                text = torch.tensor(
                    self.FILE[f"{check}/{path[1].split('/')[-1][:-4]}_{list(timings[1])}/text"][()]
                )
        return text
