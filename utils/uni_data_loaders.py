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

        max_len = 512# max([len(text.split()) for text in df[feature_col]]) + 2 # For CLS and SEP tokens
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
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
        
        try:
            df['audio_timings'] = df['audio_timings'].replace({np.nan:None})
            self.timings = df['audio_timings'].values.tolist()
        except:
            try:
                df['timings'] = df['timings'].replace({np.nan:None})
                self.timings = df['timings'].values.tolist()
                self.audio_timings = df['timings'].values.tolist()
            except:
                self.timings = [None] * len(self.audio_path)
                self.audio_timings = [None] * len(self.audio_path)

        if "meld" in dataset.lower():
            dataset = "meld"
        elif "iemo" in dataset.lower():
            dataset = "iemo"
        elif "tiktok" in dataset.lower():
            dataset = "tiktok"
        else:
            dataset = "must" if "must" in dataset.lower() else "urfunny"
            self.timings = df["timings"].values.reshape(-1, 2).tolist()
            self.audio_timings = df["audio_timings"].values.reshape(-1, 2).tolist()
            self.audio_path = df[feature_col1].values.reshape(-1, 2).tolist()
            self.video_path = df[feature_col2].values.reshape(-1, 2).tolist()
            self.texts = []
            self.text_str = []
            for i in range(0, len(df[feature_col3]), 2):

                text = df[feature_col3].iloc[i][:-1] + "</s> " + df[feature_col3].iloc[i + 1]
                self.text_str.append(text)
                # tokenize the concatenated text
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                self.texts.append(tokens)
                
            df = df[df["context"] == False]
        
        
        self.Data = Data(video=f"../../data/{dataset}_videos_blackground.hdf5",
                         audio=f"../../data/whisper_{dataset}_audio.hdf5")
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
                self.audio_path[idx], self.audio_timings[idx], self.check
            ),
            self.Data.videoMAE_features(
                self.video_path[idx], self.timings[idx], self.check
            ),
        ], np.array(self.labels[idx])
        


    
class VideoDataset(Dataset):
    """A basic dataset where the underlying data is a list of (x,y) tuples. Data
    returned from the dataset should be a (transform(x), y) tuple.
    Args:
    source      -- a list of (x,y) data samples
    transform   -- a torchvision.transforms transform
    """

    def __init__(
        self,
        df,
        dataset,
        batch_size,
        feature_col,
        label_col,
        timings=None,
        accum=False,
        check="test",
    ):
        self.video_path = df[feature_col].values

        try:
            df['timings'] = df['timings'].replace({np.nan:None})
            self.timings = df['timings'].values.tolist()
        except:
            self.timings = [None] * len(self.video_path)
        
        if "meld" in dataset and "iemo" in dataset:
            dataset = "meld_iemo"
        elif "meld" in dataset:
            dataset = "meld"
        elif "iemo" in dataset:
            dataset = "iemo"
        elif "tiktok" in dataset:
            dataset = "tiktok"
        else:
            dataset = "must" if "must" in dataset.lower() else "urfunny"
            self.timings = df["timings"].values.reshape(-1, 2).tolist()
            self.video_path = df[feature_col].values.reshape(-1, 2).tolist()
            df = df[df["context"] == False]
        
        self.Data = Data(video=f"../../data/{dataset}_videos_blackground.hdf5", audio=None)
        self.check = check
        # Make the mappings the exact same

        self.labels = df[label_col].values.tolist()
        


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
        return self.Data.videoMAE_features(
            self.video_path[idx], self.timings[idx], self.check
        ), np.array(self.labels[idx])
class WhisperDataset(Dataset):
    def __init__(
        self, df, dataset, batch_size, feature_col, label_col, accum=False, check="test"
    ):
        """
        Initialize the dataset loader.

        :data: The dataset to be loaded.
        :labels: The labels for the dataset."""

        self.audio_path = df[feature_col].values

        try:
            df['audio_timings'] = df['audio_timings'].replace({np.nan:None})
            self.timings = df['audio_timings'].values.tolist()
        except:
            try:
                df['timings'] = df['timings'].replace({np.nan:None})
                self.timings = df['timings'].values.tolist()
            except:
                self.timings = [None] * len(self.audio_path)
        
        if "meld" in dataset and "iemo" in dataset:
            dataset = "meld_iemo"
        elif "meld" in dataset:
            dataset = "meld"
        elif "iemo" in dataset:
            dataset = "iemo"
        elif "tiktok" in dataset:
            dataset = "tiktok"
        else:
            dataset = "must" if "must" in dataset.lower() else "urfunny"
            try:
                self.timings = df["audio_timings"].values.reshape(-1, 2).tolist()
            except:
                self.timings = df["timings"].values.reshape(-1, 2).tolist()
            self.audio_path = df[feature_col].values.reshape(-1, 2).tolist()
            # 0 is context, 1 is utterance
            df = df[df["context"] == False]
        
        self.Data = Data(video=None, audio=f"../../data/whisper_{dataset}_audio.hdf5")
        self.check = check
        self.labels = df[label_col].values.tolist()

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

    def __getitem__(self, idx: int):
        return self.Data.speech_file_to_array_fn(
            self.audio_path[idx], self.timings[idx], self.check
        ), np.array(self.labels[idx])

    def __len__(self):
        return len(self.labels)


class BertDataset(Dataset):
    """
    Load text dataset for BERT processing.
    """

    def __init__(self, df, dataset, batch_size, feature_col, label_col, accum=False , bert = "roberta-large"):
        
        
        
        max_len = 512# max([len(text.split()) for text in df[feature_col]]) + 2 # For CLS and SEP tokens
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
        
       
        if "meld" in dataset and "iemo" in dataset:
            dataset = "meld_iemo"
        elif "meld" in dataset:
            dataset = "meld"
        elif "iemo" in dataset:
            dataset = "iemo"
        elif "tiktok" in dataset:
            dataset = "tiktok"
        else:
            dataset = "must" if "must" in dataset.lower() else "urfunny"
            
            self.texts = []
            self.text_str = []
            for i in range(0, len(df[feature_col]), 2):

                text = df[feature_col].iloc[i][:-1] + "</s> " + df[feature_col].iloc[i + 1]
                self.text_str.append(text)
                # tokenize the concatenated text
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                self.texts.append(tokens)
            print(self.text_str[60] , flush = True)
            df = df[df["context"] == False]
            
        
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
        try:
            self.VIDEOS = (
                h5py.File(video, "r", libver="latest", swmr=True)
                if video is not None
                else None
            )
        except:
            self.VIDEOS = (
                h5py.File(video[3:], "r", libver="latest", swmr=True)
                if video is not None
                else None
            )
        try:    
            self.AUDIOS = (
                h5py.File(audio, "r", libver="latest", swmr=True)
                if audio is not None
                else None
            )
        except:
            self.AUDIOS = (
                h5py.File(audio[3:], "r", libver="latest", swmr=True)
                if audio is not None
                else None
            )
        self.must = False

        if video is not None:
            self.must = True if "must" in video or "urfunny" in video else False
            self.iemo = True if "iemo" in video else False
            self.tiktok = True if "tiktok" in video else False

        elif audio is not None:
            self.must = True if "must" in audio or "urfunny" in audio else False
            self.iemo = True if "iemo" in audio else False
            self.tiktok = True if "tiktok" in audio else False
        
        self.meld = True if (self.tiktok == False) and (self.must == False) and (self.iemo == False) else False

    def get_white_noise(self, signal: torch.Tensor, SNR , path) -> torch.Tensor:
        # @author: sleek_eagle
        shap = signal.shape
        signal = torch.flatten(signal)
        # RMS value of signal
        RMS_s = torch.sqrt(torch.mean(signal**2))
        # RMS values of noise
        RMS_n = torch.sqrt(RMS_s**2 / (pow(10, SNR / 100)))
        try:
            noise = torch.normal(0.0, RMS_n.item(), signal.shape)
        except:
            print(f"For path {path}, our signal is {signal}\n and our RMS_s is {RMS_s} \n and our RMS_n is {RMS_n}\n \n" , flush=True)
            return 0
        return noise.reshape(shap)

    def ret0(self, signal, SNR , path) -> int:
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
            
            try:
    
                video = torch.Tensor(self.VIDEOS[f"train_{path.split('/')[-1][:-4]}_{timings}"][()])  # H5PY, how to remove data after loading it into memory
            except:
                try:
                    video = torch.Tensor(self.VIDEOS[f"test_{path.split('/')[-1][:-4]}_{timings}"][()])  # H5PY, how to remove data after loading it into memory
                    
                except:
                    video = torch.Tensor(self.VIDEOS[f"val_{path.split('/')[-1][:-4]}_{timings}"][()])  # H5PY, how to remove data after loading it into memory
       
            
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
            try: # Need these 3 cases since im switching the splits for iemocap
    
                speech_array = torch.Tensor(self.AUDIOS[f"train_{path.split('/')[-1][:-4]}_{timings}"][()])
            except:
                try:
                    speech_array = torch.Tensor(self.AUDIOS[f"test_{path.split('/')[-1][:-4]}_{timings}"][()])
                    
                except:
                    speech_array = torch.Tensor(self.AUDIOS[f"val_{path.split('/')[-1][:-4]}_{timings}"][()])
       

            if check == "train":
                speech_array += singular_func_(speech_array, SNR=100 , path = path)
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
                speech_array_context += singular_func_(speech_array_context, SNR=100 , path = path)
                speech_array_target += singular_func_(speech_array_target, SNR=100 , path = path)
            # print(f"path is {path}\nshape of ret is {ret.shape}\n" , flush = True)
            return path , speech_array_target, speech_array_context , timings
