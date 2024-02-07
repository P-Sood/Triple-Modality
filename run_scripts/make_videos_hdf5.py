from argparse import ArgumentParser
import h5py
import torch
import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
import pdb
import ast
import subprocess

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)

from torchvision.transforms._transforms_video import NormalizeVideo

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)
from tqdm import tqdm

import gc
from torchvision.transforms import functional as F
import math

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


def draw(img , bbox):
    black_img = np.zeros(img.shape)
    for i in range(len(bbox)):
        x1 , y1 , x2 , y2 = bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3]
        roi = img[y1:y2 , x1:x2]
        black_img[y1:y2, x1:x2] = roi
    del img

    return black_img

def body_face(test_vid : torch.Tensor , bbox):
#   breakpoint()
  test_vid = test_vid.permute(1,0,2,3)
  bbox = ast.literal_eval(bbox) if type(bbox) == str else bbox
  for i , img in enumerate(test_vid):
    img = img.permute(1 , 2 , 0).numpy()
    output_image =  torch.Tensor(draw(img , bbox[i])).permute(2 , 0 , 1)
    del img
    test_vid[i , ...] = output_image
  return test_vid.permute(1 , 0 , 2 , 3)


def videoMAE_features(path, timings, check, speaker, bbox):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=start_time -of default=noprint_wrappers=1:nokey=1 {path}"
    start_time_offset = float(subprocess.check_output(cmd, shell=True))
    
    if timings == None:
        beg = 0
        end = 500
    else:
        beg = timings[0] - start_time_offset
        end = timings[1] - start_time_offset
        if end - beg < .1:
            beg = 0
            end = 500
        elif beg < 0 and end <0:
            beg += 2*start_time_offset
            end += 2*start_time_offset 
        elif beg < 0:
            beg = 0
            end2 = timings[1]
        elif end < 0:
            end = 500

    # feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
    # mean = feature_extractor.image_mean
    # std = feature_extractor.image_std
    # resize_to = feature_extractor.size['shortest_edge']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Need to be at 224,224 as our height and width for VideoMAE, bbox is for 224 , 224
    resize_to = {"shortest_edge": 224}
    resize_to = resize_to["shortest_edge"]
    num_frames_to_sample = 16  # SET by MODEL CANT CHANGE

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
                            #RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # Hone in on either the left_speaker or right_speaker in the video
                            Resize((resize_to, resize_to)),
                            Lambda(lambda x: x if bbox == "FAIL" else body_face(x , bbox) ), # cropped bodies only
                            
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
                            #RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)),
                            Resize((resize_to, resize_to)),
                            Lambda(lambda x: x if bbox == "FAIL" else body_face(x , bbox) ), # cropped bodies only
                        ]
                    ),
                ),
            ]
        )

    video = EncodedVideo.from_path(path, decode_audio=False)
    # breakpoint()
    video_data = video.get_clip(start_sec=beg, end_sec=end)
    try:
        video_data = transform(video_data)['video']
    except:
        try:
            video_data = video.get_clip(start_sec=beg, end_sec=end2)
            video_data = transform(video_data)['video']
        except:
            try:
                video_data = video.get_clip(start_sec=beg, end_sec=500)
                video_data = transform(video_data)['video']
            except:
                video_data = video.get_clip(start_sec=0, end_sec=500)
                video_data = transform(video_data)['video']
            
        
    del video
    del transform
    return video_data.numpy()


def write2File(writefile: h5py, path, timings, check, speaker , bbox):
    filename = f"{check}_{path.split('/')[-1][:-4]}_{timings}"
    # print(filename , flush = True)
    # generate some data for this piece of data
    if filename in writefile:
        return
    data = videoMAE_features(path[3:], timings, check, speaker , bbox)
    writefile.create_dataset(filename, data=data)
    del data
    h5py.File.flush(writefile)
    gc.collect()

def arg_parse():
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
    # pdb.set_trace()
    parser = ArgumentParser(description="Convert video into 16 frames with blackground")

    parser.add_argument(
        "--dataset",
        "-d",
        help="The dataset we are using currently, or the folder the dataset is inside",
        default="../data/urfunny.pkl",
    )

    return parser.parse_args()

def main():
    # 405 IS MESSED UP ../data/tiktok_videos/train/educative/sadboy_circus_7177431016494222638.mp4
    args = arg_parse()
    df = pd.read_pickle(args.dataset)
    df['timings'] = df['timings'].replace({np.nan:None})
    df['speaker'] = df['speaker'].replace({np.nan:None})
    if "meld" in args.dataset.lower():
        name = "meld"
    elif "iemo" in args.dataset.lower():
        name = "iemo"
    elif "must" in args.dataset.lower():
        name = "must"
    else:
        name = "urfunny"
    print(f"name is {name}")
    f = h5py.File(f"../data/{name}_videos_blackground.hdf5", "a", libver="latest", swmr=True)
    f.swmr_mode = True
    tqdm.pandas()

    df.progress_apply(
        lambda x: write2File(f, x["video_path"], x['timings'], x["split"], x['speaker'] if name == "iemo" else None , x['bbox']), axis=1
    )

    read_file = h5py.File(f"../data/{name}_videos_blackground.hdf5", "r", libver="latest", swmr=True)
    # print(list(read_file.keys()))
    print(len(list(read_file.keys())))


if __name__ == "__main__":
    main()

