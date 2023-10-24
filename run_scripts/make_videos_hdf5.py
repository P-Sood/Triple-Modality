import h5py
import torch
import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
from numpy.random import choice

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms._transforms_video import NormalizeVideo

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    PILToTensor,
)
from tqdm import tqdm

import gc
from torchvision.transforms import functional as F


class Crop:
    def __init__(self, params):
        self.params = params

    def __call__(self, frames):
        a, b, c, d = self.params
        new_vid = torch.rand((3, 16, c, d))
        # Now we are only focusing just the last two dimensions which are width and height of an image, thus we crop each frame in a video one by one
        for idx, frame in enumerate(frames):
            new_vid[idx] = F.crop(frame, *self.params)
        return new_vid


def videoMAE_features(path, timings, check, speaker):
    # TODO: DATASET SPECIFIC
    if timings == None:
        beg = 0
        end = 500
    else:
        beg = timings[0]
        end = timings[1]
        if end - beg < 0.1:
            beg = 0
            end = 500

    # feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
    # mean = feature_extractor.image_mean
    # std = feature_extractor.image_std
    # resize_to = feature_extractor.size['shortest_edge']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize_to = {"shortest_edge": 224}
    resize_to = resize_to["shortest_edge"]
    num_frames_to_sample = 8  # SET by MODEL CANT CHANGE

    # print(f"We have path {path}\ntimings are {timings}\nspeaker is {speaker}\ncheck is {check}\nsingular_func_ is {singular_func_.__name__ if singular_func_ is not None else None}" , flush= True)

    # i hardcoded the feature extractor stuff just because it saves a couple milliseconds every run, so hopefully makes
    # it a tad faster overall
    if check == "train":
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            NormalizeVideo(mean, std),
                            # RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # Hone in on either the left_speaker or right_speaker in the video
                            Resize(
                                (resize_to, resize_to)
                            ),  # Need to be at 224,224 as our height and width for VideoMAE, bbox is for 224 , 224
                        ]
                    ),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            NormalizeVideo(mean, std),
                            # RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )

    video = EncodedVideo.from_path(path, decode_audio=False)
    video_data = video.get_clip(start_sec=186, end_sec=end)
    del video
    video_data = transform(video_data)
    del transform
    return video_data["video"].numpy()


def write2File(writefile: h5py, path, timings, check, speaker):
    print(f"{path}", flush=True)
    # filename = f"{check}_{path.split('/')[-1][:-4]}_{timings}"
    filename = f"{check}_{path.split('/')[-1][:-4]}_2"
    # print(filename , flush = True)
    # generate some data for this piece of data
    data = videoMAE_features(path, timings, check, speaker)
    writefile.create_dataset(filename, data=data)
    del data
    h5py.File.flush(writefile)
    gc.collect()


def fun(df, f, i):
    sub_df = df.iloc[i * 1 : (i + 1) * 1]
    sub_df.apply(
        lambda x: write2File(f, x["video_path"], None, x["split"], None), axis=1
    )


def main():
    # 405 IS MESSED UP ../data/tiktok_videos/train/educative/sadboy_circus_7177431016494222638.mp4
    a = 405
    df = pd.read_pickle("/home/jupyter/multi-modal-emotion/data/tiktok.pkl")[a : a + 1]
    print(df["video_path"])
    f = h5py.File("../data/tiktok_videos.hdf5", "a", libver="latest", swmr=True)
    f.swmr_mode = True
    # tqdm.pandas()
    for i in tqdm(
        range(0, 1)
    ):  # change first arg in range, to the next applicable one in case it crashes
        fun(df, f, i)
        gc.collect()

    # df.apply(lambda x: write2File(f , x['video_path'] , None , x['split'] , None  ) , axis = 1 )

    read_file = h5py.File("../data/tiktok_videos.hdf5", "r", libver="latest", swmr=True)
    print(list(read_file.keys()))
    print(len(list(read_file.keys())))


if __name__ == "__main__":
    main()
