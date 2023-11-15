import gc
import h5py
import pandas as pd
import whisper
import math
from tqdm import tqdm



def speech_file_to_array_fn(path, timings, speaker, target_sampling_rate=16000):
    speech_array = whisper.load_audio(path , target_sampling_rate) # Loads it in as a numpy array
    sampling_rate = target_sampling_rate
    if speaker is None and timings is not None:  # MELD is only with Speaker is None and timings is not None, since speaker is not a part of meld pkl
        start = timings[0]
        end = timings[1]
        if end - start > 0.2:
            start_sample = math.floor(start * sampling_rate)
            end_sample = math.ceil(end * sampling_rate)
            # extract the desired segment
            if end_sample - start_sample > sampling_rate*0.2:
                speech_array = speech_array[start_sample:end_sample]

    del sampling_rate
    return speech_array


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
    df = pd.read_pickle("../data/meld.pkl")
    f = h5py.File("../data/whisper_meld_audio.hdf5", "a", libver="latest", swmr=True)
    f.swmr_mode = True
    tqdm.pandas()
    df.progress_apply(
        lambda x: write2File(f, x["audio_path"], x['timings'], None, x["split"]), axis=1
    )
    read_file = h5py.File("../data/whisper_meld_audio.hdf5", "r", libver="latest", swmr=True)
    print(len(list(read_file.keys())))


if __name__ == "__main__":
    main()
