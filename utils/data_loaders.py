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

        
        if timings != None:
            try:
                self.timings = df[timings].values.tolist()
            except: 
                self.timings = [None] * len(self.audio_path)
        else:
            self.timings = [None] * len(self.audio_path)

        if "meld" in str(dataset).lower():
            max_len = int(70 * 2.5)
            tokenizer = AutoTokenizer.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            )
            self.Data = Data(
                video="../../data/videos_context.hdf5", audio="../../data/audio.hdf5"
            )
            self.texts = [
                tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                for text in df[feature_col3]
            ]
        elif "iemo" in str(dataset).lower():
            max_len = int(70 * 2.5)
            tokenizer = AutoTokenizer.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            )
            self.Data = Data(
                video="../../data/iemo_videos.hdf5", audio="../../data/iemo_audio.hdf5"
            )
            self.texts = [
                tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                for text in df[feature_col3]
            ]
        elif "tiktok" in str(dataset).lower():
            max_len = 300
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            self.Data = Data(
                video="../../data/tiktok_videos.hdf5",
                audio="../../data/tiktok_audio.hdf5",
            )
            self.texts = [
                tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                for text in df[feature_col3]
            ]
        else:
            max_len = 300
            tokenizer = AutoTokenizer.from_pretrained(
                "jkhan447/sarcasm-detection-RoBerta-base-CR"
            )
            self.Data = Data(
                video="../../data/must_videos.hdf5", audio="../../data/must_audio.hdf5"
            )
            self.texts = []
            for i in range(0, len(df[feature_col2]), 2):
                # concatenate the text
                text = df[feature_col3].iloc[i] + " " + df[feature_col3].iloc[i + 1]
                # tokenize the concatenated text
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                self.texts.append(tokens)
            self.timings = df["timings"].values.reshape(-1, 2).tolist()
            self.audio_path = df[feature_col1].values.reshape(-1, 2).tolist()
            self.video_path = df[feature_col2].values.reshape(-1, 2).tolist()
            df = df[df["context"] == False]
        self.check = check
        self.labels = df[label_col].values.tolist()

        assert (
            len(self.audio_path) == len(self.video_path) == len(self.texts)
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
        return [
            self.texts[idx],
            self.Data.speech_file_to_array_fn(
                self.audio_path[idx], self.timings[idx], self.check
            ),
            self.Data.videoMAE_features(
                self.video_path[idx], self.timings[idx], self.check
            ),
        ], np.array(self.labels[idx])



class BertDataset(Dataset):
    """
    Load text dataset for BERT processing.
    """

    def __init__(self, df, dataset, batch_size, feature_col, label_col, accum=False , bert = "j-hartmann/emotion-english-distilroberta-base"):
        
        max_len = int(70 * 2.5)
        tokenizer = AutoTokenizer.from_pretrained(bert)
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            for text in df[feature_col]
        ]
        
        self.labels = df[label_col].values.tolist()
        
        assert len(self.texts) == len(self.labels), f"Got a mismatch texts: {len(self.texts)} != labels:{len(self.labels)}"

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
        return self.texts[idx], self.labels[idx]


class Data:
    def __init__(self, video, audio) -> None:
        self.VIDEOS = (
            h5py.File(video, "r", libver="latest", swmr=True)
            if video is not None
            else None
        )
        self.AUDIOS = (
            h5py.File(audio, "r", libver="latest", swmr=True)
            if audio is not None
            else None
        )
        self.must = False

        if video is not None:
            self.must = True if "must" in video else False
            self.iemo = True if "iemo" in video else False
            self.tiktok = True if "tiktok" in video else False

        elif audio is not None:
            self.must = True if "must" in audio else False
            self.iemo = True if "iemo" in audio else False
            self.tiktok = True if "tiktok" in audio else False
        
        self.meld = True if (self.tiktok == False) and (self.must == False) and (self.iemo == False) else False

    def get_white_noise(self, signal: torch.Tensor, SNR) -> torch.Tensor:
        # @author: sleek_eagle
        shap = signal.shape
        signal = torch.flatten(signal)
        # RMS value of signal
        RMS_s = torch.sqrt(torch.mean(signal**2))
        # RMS values of noise
        RMS_n = torch.sqrt(RMS_s**2 / (pow(10, SNR / 100)))
        noise = torch.normal(0.0, RMS_n.item(), signal.shape)
        return noise.reshape(shap)

    def ret0(self, signal, SNR) -> int:
        return 0

    def videoMAE_features(self, path, timings, check):
        if check == "train":
            transform = Compose(
                [
                    RandomHorizontalFlip(p=0.5),  #
                    RandomVerticalFlip(p=0.5),  #
                ]
            )
        else:
            transform = Compose(
                [
                    RandomHorizontalFlip(p=0),
                ]
            )

        if not self.must:
            if self.meld: # MELD
                video = torch.Tensor(
                    self.VIDEOS[f"{check}_{path.split('/')[-1][:-4]}"][()]
                )  # H5PY, how to remove data after loading it into memory
                video = transform(video)
                return path , video, timings
            else: #IEMO OR TIKTOK
                video = torch.Tensor(
                    self.VIDEOS[f"{check}_{path.split('/')[-1][:-4]}_{timings}"][()]
                )  # H5PY, how to remove data after loading it into memory
                video = transform(video)

                return path , video, timings
        else:
            #TODO: Check which path is to which place. Make sure context and targets are right pathing
            video_context = torch.Tensor(
                self.VIDEOS[f"{check}_{path[0].split('/')[-1][:-4]}_{timings[0]}"][()]
            )  # H5PY, how to remove data after loading it into memory
            video_target = torch.Tensor(
                self.VIDEOS[f"{check}_{path[1].split('/')[-1][:-4]}_{timings[1]}"][()]
            )  # H5PY, how to remove data after loading it into memory
            video_context = transform(video_context)
            video_target = transform(video_target)

            return path , video_target, video_context, timings

    def speech_file_to_array_fn(self, path, timings, check="train"):
        func_ = [self.ret0, self.get_white_noise]
        singular_func_ = random.choices(population=func_, weights=[0.5, 0.5], k=1)[0]

        if not self.must:
            if self.meld: # MELD
                speech_array = torch.Tensor(self.AUDIOS[f"{check}_{path.split('/')[-1][:-4]}"][()])
            else: # IEMO OR TIKTOK
                speech_array = torch.Tensor(self.AUDIOS[f"{check}_{path.split('/')[-1][:-4]}_{timings}"][()])

            if check == "train":
                speech_array += singular_func_(speech_array, SNR=10)
            # print(f"path is {path}\nshape of ret is {ret.shape}\n" , flush = True)
            return path , speech_array , timings
        else:
            speech_array_context = torch.Tensor(
                self.AUDIOS[f"{check}_{path[0].split('/')[-1][:-4]}_{timings[0]}"][()]
            )
            speech_array_target = torch.Tensor(
                self.AUDIOS[f"{check}_{path[1].split('/')[-1][:-4]}_{timings[1]}"][()]
            )

            if check == "train":
                speech_array_context += singular_func_(speech_array_context, SNR=10)
                speech_array_target += singular_func_(speech_array_target, SNR=10)
            # print(f"path is {path}\nshape of ret is {ret.shape}\n" , flush = True)
            return path , speech_array_target, speech_array_context , timings
