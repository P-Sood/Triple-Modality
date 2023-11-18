import h5py
def audio():
    af1 = 'data/whisper_meld_audio.hdf5'
    af2 = 'data/whisper_iemo_audio.hdf5'
    output_af = 'data/whisper_meld_iemo_audio.hdf5'

    with h5py.File(af1, 'r' , libver="latest", swmr=True) as f1, h5py.File(af2, 'r' , libver="latest", swmr=True) as f2, h5py.File(output_af, 'w' , libver="latest", swmr=True) as fo:
        # Iterate over datasets in the first file
        for name, data in f1.items():
            # Copy each dataset to the output file
            f1.copy(name, fo)

        # Iterate over datasets in the second file
        for name, data in f2.items():
            # Copy each dataset to the output file
            f2.copy(name, fo)

if __name__ == "__main__":
    audio()
