import h5py
import torch
import pandas as pd
import torchaudio
import math


from tqdm import tqdm

import gc


def speech_file_to_array_fn(path, timings, speaker, target_sampling_rate=16000):
    speech_array, sampling_rate = torchaudio.load(path)
    if speaker is None and timings is not None:  # MELD
        start = timings[0]
        end = timings[1]
        if end - start > 0.2:
            start_sample = math.floor(start * sampling_rate)
            end_sample = math.ceil(end * sampling_rate)
            # extract the desired segment
            if end_sample - start_sample > 3280:
                speech_array = speech_array[:, start_sample:end_sample]

    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    del sampling_rate
    speech_array = resampler(speech_array).squeeze()
    speech_array = (
        torch.mean(speech_array, dim=0) if len(speech_array.shape) > 1 else speech_array
    )  # over the channel dimension
    return speech_array.numpy()


def write2File(writefile, path, timings, speaker, check):
    filename = f"{check}_{path.split('/')[-1][:-4]}_{timings}"
    # generate some data for this piece of data
    data = speech_file_to_array_fn(path, timings, speaker)
    # ERRORS HERE BECAUSE OF PATHING
    writefile.create_dataset(filename, data=data)
    gc.collect()


def fun(df, f, i):
    sub_df = df.iloc[i * 1000 : (i + 1) * 1000]
    sub_df.progress_apply(
        lambda x: write2File(f, x["audio_path"], None, None, x["split"]), axis=1
    )


def main():
    df = pd.read_pickle("../data/tiktok_sample.pkl")
    f = h5py.File("../data/tiktok_audio.hdf5", "a", libver="latest", swmr=True)
    f.swmr_mode = True
    tqdm.pandas()
    df.progress_apply(
        lambda x: write2File(f, x["audio_path"], None, None, x["split"]), axis=1
    )
    read_file = h5py.File("../data/tiktok_audio.hdf5", "r", libver="latest", swmr=True)
    print(len(list(read_file.keys())))


if __name__ == "__main__":
    main()
