"""
File was created using https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
It will get the timestamp for each character, and from there figure out the timestamp for each word, to sentence etc.
It works better for shorter text, but havent dealt with many super huge pieces of text/audio files yet

You might encounter some bugs with the text itself, because wav2vec2 expects it to be a certain way, thus i have created multiple functions
that should allow you to take in the raw text, and from there it will make replacements to remove certain characters or go from numerals to 
the text of a numeral (4 -> four)

The output text will be something like

"Hey, Bob!" -> "HEY|BOB"

"""
from tqdm import tqdm
from num2words import num2words
import numpy as np
import pandas as pd
import re
import os
from dataclasses import dataclass
import pandas as pd
import torch
import torchaudio
import string
from argparse import ArgumentParser
from unidecode import unidecode
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from spleeter.separator import Separator


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def replace_ordinal_numbers(text):
    re_results = re.findall("(\d+(st|nd|rd|th))", text)
    for enitre_result, suffix in re_results:
        num = int(enitre_result[: -len(suffix)])
        text = text.replace(enitre_result, num2words(num, ordinal=True))
    return text


def replace_numbers(text):
    re_results = re.findall("\d+", text)
    for term in re_results:
        num = int(term)
        text = text.replace(term, num2words(num))
    return text


def convert_numbers(text):
    text = replace_ordinal_numbers(text)
    text = replace_numbers(text)

    return text


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When refering to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when refering to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))
        del prob

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        """This is usually done with smaller wav files, play around here if youd like something else to occur here"""
        return None
        # raise ValueError('Failed to align')
    del stayed
    del changed
    return path[::-1]


def merge_repeats(path, ratio, transcript, sr):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return (segments[0].start * ratio) / sr, (segments[-1].end * ratio) / sr


def general_backtrack(text, size, sample_rate, emission, labels):
    rep = {
        ",": "",
        "!": "",
        '"': "",
        "#": "",
        "$": "",
        "%": "",
        "&": "",
        "(": "",
        ")": "",
        "*": "",
        "+": "",
        ".": "",
        "/": "",
        ":": "",
        ";": "",
        "<": "",
        "=": "",
        ">": "",
        "?": "",
        "@": "",
        "[": "",
        "\\": "",
        "]": "",
        "^": "",
        "_": "",
        "`": "",
        "{": "",
        "}": "",
        "~": "",
        "'": "",
        "’": "",
        "`": "",
        "-": "",
        "‘": "",
        "—": "",
        "…": "",
        "…": "",
        "…": "",
    }

    # rep just becomes the same dictionary, but with support for re functions
    non_alphanumeric_chars = string.punctuation
    my_dict = {char: "" for char in non_alphanumeric_chars}
    rep = {**rep, **my_dict}
    rep = dict((re.escape(k), v) for k, v in rep.items())

    pattern = re.compile("|".join(rep.keys()))

    # This is where we capitalize, and remove all punctuation and characters from rep
    transcript = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.upper())
    # Split the transcript based on spaces
    transc_list = transcript.split(" ")

    # Here is where we convert 4 -> FOUR
    for i, word in enumerate(transc_list):
        if word.isnumeric() or any(map(str.isdigit, word)):
            transc_list[i] = (
                convert_numbers(word)
                .replace("-", " ")
                .replace(",", " ")
                .replace(" ", "|")
                .upper()
            )

    transcript = "|".join(transc_list)
    transcript = transcript.replace(" ", "|")

    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]

    trellis = get_trellis(emission, tokens)

    path = backtrack(trellis, emission, tokens)
    del emission
    if path == None:  # This can happen if wav files are too short
        return None
    ratio = size / (trellis.size(0) - 1)
    timings = merge_repeats(path, ratio, transcript, sample_rate)

    return timings


MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(
    "cuda"
)
PROC = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
SEP = Separator("spleeter:2stems")


def get_times(path, model_cuda, text, labels, sample_rate, enable_gpu):
    text = unidecode(text)
    with torch.inference_mode():
        waveform, _ = torchaudio.load(path[3:])
        print(waveform.shape, flush=True)
        waveform = SEP.separate(
            waveform.T
        )  # This is where we separate the vocals from the audio, using spleeter
        print(waveform, flush=True)
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        size = waveform[0].size(0)
        if enable_gpu and waveform.shape[1] < 3000000 and text is not "[N/A]":
            emissions, _ = model_cuda(waveform.to("cuda"))
            emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()
            del emissions

            timings = general_backtrack(text, size, sample_rate, emission, labels)
            del emission
            return timings
        else:
            first_part = waveform[:, :1000000]
            last_part = waveform[:, -1000000:]
            parts = [first_part, last_part]
            timings = [0, 0]
            for enum, part in enumerate(parts):
                input_values = PROC(
                    part, sampling_rate=sample_rate, return_tensors="pt"
                ).input_values
                emissions, _ = model_cuda(part.to("cuda"))

                del part

                logits = MODEL(input_values.squeeze(0).to("cuda")).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transc = PROC.batch_decode(predicted_ids)[0]

                emissions = torch.log_softmax(emissions, dim=-1)
                emission = emissions[0].cpu().detach()
                del emissions

                timings[enum] = general_backtrack(
                    transc, size, sample_rate, emission, labels
                )
                del emission
            return (timings[0][0], timings[1][1])


def arg_parse():
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
    parser = ArgumentParser(
        description=f"Get timings of text, based on accompanied wav file"
    )

    parser.add_argument(
        "--dataset",
        "-d",
        help="The dataset we are using currently",
        default="../data/tiktok.pkl",
        type=str,
    )
    parser.add_argument(
        "--enable_gpu", "-e", help="Use a GPU or not", default=True, type=bool
    )
    return parser.parse_args()


def fun(df, dataset, i):
    sub_df = df.iloc[i * 40 : (i + 1) * 40]
    sub_df["timings"] = sub_df.progress_apply(
        lambda x: get_times(
            x["audio_path"],
            model_cuda,
            x["text"].split(":")[-1][1:],
            labels,
            sr,
            enable_gpu,
        ),
        axis=1,
    )
    sub_df.to_pickle(
        f"{dataset[:-4]}_{i}_new.pkl"
    )  # change this to the next applicable one in case it crashes


if __name__ == "__main__":
    args = arg_parse()
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    dataset = args.dataset
    enable_gpu = args.enable_gpu

    df = pd.read_pickle(dataset[3:])

    bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
    model_cuda = bundle.get_model().to("cuda")

    labels = bundle.get_labels()
    sr = bundle.sample_rate

    tqdm.pandas()
    for i in tqdm(
        range(0, 25)
    ):  # change first arg in range, to the next applicable one in case it crashes
        fun(df, dataset, i)

    # df.to_pickle(dataset)
