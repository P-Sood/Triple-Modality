
import os
import sys
sys.path.insert(0,"/home/zeerak.talat/trimodal/") 
__package__ = 'run_scripts'
# Add the OpenCV library path to the system path
# sys.path.append('/apps/local/opencv-gpu/lib')
# sys.path.append('/apps/local/opencv/lib')
sys.path.append('/apps/local/opencv-gpu/lib/python3.8/site-packages/')

# Add the OpenCV binary path to the system path
# os.environ["PATH"] += os.pathsep + '/apps/local/opencv-gpu/bin'
# os.environ["PATH"] += os.pathsep + '/apps/local/opencv/bin'

from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from utils.global_functions import Crop
from argparse import ArgumentParser
# import dlib
# import cv2
# DETECTOR = dlib.cnn_face_detection_model_v1("/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/mmod_human_face_detector.dat")

# from facenet_pytorch import MTCNN
# MODEL = MTCNN(keep_all=True, device="cuda")
import cvlib

def arg_parse():
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
    parser = ArgumentParser(description= f"Get body bounding box from video")

    parser.add_argument("--dataset"  , "-d", help="The dataset we are using currently", default = "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/MUSTARD_face_body.pkl" , type=str) 
    parser.add_argument("--enable_gpu" , "-e" , help="Use a GPU or not"  , default=False, type=bool)
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
    
    print(f"We are using dataset {dataset}")


    df = pd.read_pickle(dataset)
    if "IEMOCAP" in dataset or "IEMO" in dataset:
        name = "IEMOCAP"
        bg = 0
        rn = 8
    elif "MELD" in dataset or "TAV" in dataset:
        name = "MELD"
        bg = 0
        rn = 14
    elif "must" in dataset:
        name = "MUSTARD"
        bg = 0
        rn = 3
    elif "ur" in dataset:
        name = "URFUNNY"
        bg = 0
        rn = 21
    

    tqdm.pandas()
    print("Starting progress Apply")
    
    for i in range(bg,rn): # change first arg in range, to the next applicable one in case it crashes
        print(f"Last finished dataset was {i}" , flush = True)
        sub_df = df.iloc[i*1000:(i+1)*1000]
        # sub_df['bbox'] = sub_df.progress_apply(lambda x: boxes(x['video_path'] , x['timings'] , x['speaker'] ), axis = 1 )
        sub_df["bbox"]  = sub_df.progress_apply(lambda x: boxes(x['video_path'] , x['video_timings'] , x['speaker'] , enable_gpu ) , axis = 1 )
        sub_df.to_pickle(f"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/{name}_df_sub_{i}.pkl")
    
def body_face(test_vid:torch.Tensor , enable_gpu: bool):
  test_vid = test_vid.permute(1,0,2,3)
  faces = []
  boxes = []
  for img in test_vid:
    img = np.ascontiguousarray(img.permute(1 , 2 , 0))
    
    # img = img.astype(np.uint8)
    # face, _ = MODEL.detect(img)
    # faces.append(face)
    
    # results = DETECTOR(img , 2)
    # face =  [convert_and_trim_bb(img, r.rect) for r in results]
    # faces.append(face)
    
    bbox , label , _ = cvlib.detect_common_objects(img , confidence=0.5 , model = 'yolov4' , enable_gpu=enable_gpu)
    bbox = np.array(bbox)[np.array(label) == "person"].tolist()
    boxes.append(bbox)
  return  boxes
    
def boxes(path  , clip_duration , speaker , enable_gpu ):
    # TODO: DATASET SPECIFIC
    if clip_duration == None:
        beg = 0
        end = 500
    else:
        beg = clip_duration[0]
        end = clip_duration[1]
        if end - beg < .1:
            beg = 0
            end = 500


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
    video = EncodedVideo.from_path(path[3:]) # Due to folder stuff
    
    video_data = video.get_clip(start_sec=beg, end_sec=end)
    video_data = transform(video_data)['video']
     
    return body_face(video_data , enable_gpu)
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    # pip install pandas;pip install numpy; pip install tqdm; pip install torch; pip install pytorchvideo; pip install transformers; pip install cvlib; pip install torchvision; pip install tensorflow