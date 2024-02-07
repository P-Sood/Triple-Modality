
import os
import sys
sys.path.insert(0,"/home/zeerak.talat/trimodal/") 
__package__ = 'run_scripts'
sys.path.append('/apps/local/opencv-gpu/lib/python3.8/site-packages/')


import subprocess
import gc
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import cvlib

from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from argparse import ArgumentParser

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)

# import dlib
# import cv2
# DETECTOR = dlib.cnn_face_detection_model_v1("/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/mmod_human_face_detector.dat")

# from facenet_pytorch import MTCNN
# MODEL = MTCNN(keep_all=True, device="cuda")

import pdb
def arg_parse():
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
    parser = ArgumentParser(description= f"Get body bounding box from video")

    parser.add_argument("--dataset"  , "-d", help="The dataset we are using currently", default = "../data/must.pkl" , type=str) 
    parser.add_argument("--enable_gpu" , "-e" , help="Use a GPU or not"  , default=False, type=bool)
    parser.add_argument("--beg" , "-b" , help="beginning"  , default=False, type=int)
    parser.add_argument("--end" , "-en" , help="ending"  , default=False, type=int)
    return parser.parse_args()

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# return our bounding box coordinates
	return [startX, startY, endX, endY]


# Create an MTCNN face detector instance


def main():
    args = arg_parse()
    
    dataset = args.dataset
    enable_gpu = args.enable_gpu
    beg = args.beg
    end = args.end
    
    print(f"args:  {args}")


    df = pd.read_pickle(dataset)
    if "IEMOCAP" in dataset or "IEMO" in dataset:
        name = "IEMOCAP"
        bg = beg
        rn = end
    elif "MELD" in dataset or "TAV" in dataset:
        name = "MELD"
        bg = beg
        rn = end
    elif "must" in dataset:
        name = "MUSTARD"
        bg = beg
        rn = end
    elif "ur" in dataset:
        name = "URFUNNY"
        bg = beg
        rn = end
        
    print(f"we are in range {bg} to {rn}" , flush = True)
    tqdm.pandas()
    for i in range(bg,rn): # change first arg in range, to the next applicable one in case it crashes
        print(f"Last finished dataset was {i}" , flush = True)
        sub_df = df.iloc[i*1000:(i+1)*1000]
        # sub_df['bbox'] = sub_df.progress_apply(lambda x: boxes(x['video_path'] , x['timings'] , x['speaker'] ), axis = 1 )
        sub_df["bbox"]  = sub_df.progress_apply(lambda x: boxes(x['video_path'] , x['video_timings'] , x['speaker'] , enable_gpu ) , axis = 1 )
        sub_df.to_pickle(f"/home/zeerak.talat/trimodal/data/{name}_df_sub_{i}.pkl")
    
def body_face(test_vid:torch.Tensor , enable_gpu: bool):
    test_vid = test_vid.permute(1,0,2,3)
    boxes = []
    for img in test_vid:
        img = np.ascontiguousarray(img.permute(1 , 2 , 0))
        bbox , label , _ = cvlib.detect_common_objects(img , confidence=0.5 , model = 'yolov4-tiny' , enable_gpu=enable_gpu)
        bbox = np.array(bbox)[np.array(label) == "person"].tolist()
        boxes.append(bbox)
        gc.collect()
    del test_vid
    return  boxes
    
def boxes(path  , clip_duration , speaker , enable_gpu ):
    # TODO: DATASET SPECIFIC
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=start_time -of default=noprint_wrappers=1:nokey=1 {path[3:]}"
    start_time_offset = float(subprocess.check_output(cmd, shell=True))

    if clip_duration == None:
        beg = 0
        end = 500
    else:
        beg = clip_duration[0] - start_time_offset
        end = clip_duration[1] - start_time_offset
        if end - beg < .1:
            beg = 0
            end = 500
        elif beg < 0 and end <0:
            beg += 2*start_time_offset
            end += 2*start_time_offset 
        elif beg < 0:
            beg = 0
            end2 = clip_duration[1]
        elif end < 0:
            end = 500
    del start_time_offset

    resize_to = {'shortest_edge': 224}
    resize_to = resize_to['shortest_edge']
    num_frames_to_sample = 16 # SET by MODEL CANT CHANGE
    
    
    transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        # RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # Switch on or off For dataset
                        Resize((224, 224)), 
                    ]
                ),
            ),
        ]
    )
    video = EncodedVideo.from_path(path[3:] , decode_audio=False) # Due to folder stuff
    try:
        video_data = video.get_clip(start_sec=beg, end_sec=end)
        video_data = transform(video_data)['video']
        del video
    except:
        try:
            video_data = video.get_clip(start_sec=beg, end_sec=end2)
            video_data = transform(video_data)['video']
            del video
        except:    
            del video
            return "FAIL"
    del transform


    ret = body_face(video_data , enable_gpu)
    del video_data
    gc.collect()
    return ret
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    # pip install --upgrade pip; pip install pandas;pip install numpy; pip install tqdm; pip install torch; pip install pytorchvideo; pip install transformers; pip install cvlib; pip install torchvision; pip install tensorflow