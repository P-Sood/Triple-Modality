from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import gc
import h5py
import pandas as pd
import whisper
import math
from tqdm import tqdm
import ast
import numpy as np

def speech_file_to_array_fn(path, timings, target_sampling_rate=16000):
    speech_array = whisper.load_audio(path , target_sampling_rate) # Loads it in as a numpy array
    sampling_rate = target_sampling_rate
    if timings is not None:  # MELD is only with Speaker is None and timings is not None, since speaker is not a part of meld pkl
        # try:
        #     timings = ast.literal_eval(timings)
        # except:
        #     pass
        start = timings[0]
        end = timings[1]
        if end - start > 0.2:
            start_sample = math.floor(start * sampling_rate)
            end_sample = math.ceil(end * sampling_rate)
            # extract the desired segment
            if start_sample >= len(speech_array):
                speech_array = speech_array
            elif (end_sample >= len(speech_array)) and (start_sample < len(speech_array)):
                speech_array = speech_array[start_sample:]
            else:
                speech_array = speech_array[start_sample:end_sample]
    if len(speech_array) < 10:
        print("Error with file: ", path , "speech_array is too small, at length: ", len(speech_array) , flush=True)
    del sampling_rate
    return speech_array


def write2File(writefile, path, timings, check):
    filename = f"{check}_{path.split('/')[-1][:-4]}_{timings}"
    # generate some data for this piece of data
    data = speech_file_to_array_fn(path[3:], timings)
    # ERRORS HERE BECAUSE OF PATHING
    writefile.create_dataset(filename, data=data)
    gc.collect()


def fun(df, f, i):
    sub_df = df.iloc[i * 1000 : (i + 1) * 1000]
    sub_df.progress_apply(
        lambda x: write2File(f, x["audio_path"], x['timings'], x["split"]), axis=1
    )

def arg_parse():
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
    # pdb.set_trace()
    parser = ArgumentParser(description="Convert raw audio to whisper audio")

    parser.add_argument(
        "--dataset",
        "-d",
        help="The dataset we are using currently, or the folder the dataset is inside",
        default="../data/meld.pkl",
    )

    return parser.parse_args()

def main():
    args = arg_parse()
    df = pd.read_pickle(args.dataset)
    df['timings'] = df['timings'].replace({np.nan:None})
    if "ur" in args.dataset.lower():
        name = "urfunny"
    elif "iemo" in args.dataset.lower():
        name = "iemo"
    else:
        name = "must"
    f = h5py.File(f"../data/whisper_{name}_audio.hdf5", "a", libver="latest", swmr=True)
    f.swmr_mode = True
    tqdm.pandas()
    df.progress_apply(
        lambda x: write2File(f, x["audio_path"], x['audio_timings'], x["split"]), axis=1
    )
    read_file = h5py.File(f"../data/whisper_{name}_audio.hdf5", "r", libver="latest", swmr=True)
    print(len(list(read_file.keys())))


if __name__ == "__main__":
    main()