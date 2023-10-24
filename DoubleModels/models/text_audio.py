import torch
from torch import nn
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModel
import torchaudio
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch, must):  # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    text_mask = []

    input_list = []
    speech_list = []
    label_list = []
    text_list_mask = None
    speech_list_mask = None

    speech_context = []
    speech_list_context_input_values = torch.empty((1))

    for input, label in batch:
        text = input[0]
        input_list.append(text["input_ids"].tolist()[0])
        text_mask.append(text["attention_mask"].tolist()[0])
        audio_path = input[1]
        if not must:
            speech_list.append(audio_path)
        else:
            speech_list.append(audio_path[0])
            speech_context.append(audio_path[1])

        label_list.append(label)

    text_list_mask = torch.Tensor(np.array(text_mask))

    del text_mask

    speech_list_input_values = pad_sequence(
        speech_list, batch_first=True, padding_value=0
    )
    del speech_list

    if must:
        speech_list_context_input_values = pad_sequence(
            speech_context, batch_first=True, padding_value=0
        )
        del speech_context

    text = {
        "input_ids": torch.Tensor(np.array(input_list)).type(torch.LongTensor),
        "attention_mask": text_list_mask,
    }

    audio_features = {
        "audio_features": speech_list_input_values,
        "attention_mask": speech_list_mask,
        "audio_context": speech_list_context_input_values,
    }

    return [text, audio_features], torch.Tensor(np.array(label_list))


class BertAudioClassifier(nn.Module):
    def __init__(self, args):
        super(BertAudioClassifier, self).__init__()
        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.learn_PosEmbeddings = args["learn_PosEmbeddings"]
        self.num_layers = args["num_layers"]
        self.dataset = args["dataset"]

        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.6

        if self.must:
            self.bert = AutoModel.from_pretrained(
                "jkhan447/sarcasm-detection-RoBerta-base-CR"
            )
        else:
            self.bert = AutoModel.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            )

        self.test_ctr = 1
        self.train_ctr = 1

        self.wav2vec2 = AutoModel.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.wav_2_768_2 = nn.Linear(1024, 768)
        self.aud_norm = nn.LayerNorm(1024)

        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)

        self.bert_norm = nn.LayerNorm(768)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768 * 2, self.output_dim)

    def forward(
        self, input_ids, text_attention_mask, audio_features, context_audio, check
    ):
        _, text_outputs = self.bert(
            input_ids=input_ids, attention_mask=text_attention_mask, return_dict=False
        )
        del _
        del input_ids
        del text_attention_mask
        text_outputs = self.bert_norm(text_outputs)  #

        aud_outputs = self.wav2vec2(audio_features)[0]
        del audio_features
        aud_outputs = torch.mean(aud_outputs, dim=1)
        aud_outputs = self.aud_norm(aud_outputs)
        if self.must:
            aud_context = self.wav2vec2(context_audio)[0]
            del context_audio
            aud_context = torch.mean(aud_context, dim=1)
            aud_context = self.aud_norm(aud_context)
            aud_outputs = (aud_outputs * self.p + aud_context * (1 - self.p)) / 2
            del aud_context

        aud_outputs = self.wav_2_768_2(aud_outputs)

        ta = torch.cat([text_outputs, aud_outputs], dim=1)

        del text_outputs
        del aud_outputs

        # Classifier Head
        if check == "train":
            ta = self.dropout(ta)
        ta = self.linear1(ta)

        return ta  # returns [batch_size,output_dim]
