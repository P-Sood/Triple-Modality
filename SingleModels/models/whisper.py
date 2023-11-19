import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperForAudioClassification, WhisperFeatureExtractor
import pdb


FEAT = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

def collate_batch(batch, must):  # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    speech_list = []
    speech_list_context = []
    label_list = []
    speech_list_context_input_values = torch.empty((1))
    #breakpoint()
    for input, label in batch:
        if not must:
            speech_list.append(FEAT(input[1] , sampling_rate=16000).input_features[0])
        else:
            speech_list.append(FEAT(input[1], sampling_rate=16000).input_features[0])
            speech_list_context.append(FEAT(input[2], sampling_rate=16000).input_features[0])

        label_list.append(label)

    if must:
        speech_list_context_input_values = torch.Tensor(np.array(speech_list_context))

    audio_features = {
        "audio_features": torch.Tensor(np.array(speech_list)),
        "context_audio": speech_list_context_input_values,
    }

    return audio_features, torch.Tensor(np.array(label_list)).long()


class WhisperForEmotionClassification(nn.Module):
    """_summary_

    Args:
        args: dictionary filled with output_dim, dropout, dataset
        
        Returns:
            aud_outputs: tensor of shape [batch_size, output_dim] logits for each class
    """
    def __init__(self, args):
        super(WhisperForEmotionClassification, self).__init__()
        self.output_dim = args["output_dim"]
        self.dropout = args["dropout"]
        self.dataset = args["dataset"]

        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.6

        self.whisper = WhisperForAudioClassification.from_pretrained("openai/whisper-medium")

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

    def forward(self, audio_features, context_audio, check):
        aud_outputs = self.whisper.encoder(audio_features)[0]
        aud_outputs = aud_outputs.mean(dim=1)
        del audio_features
        if self.must:
            aud_context = self.whisper(context_audio).hidden_states.mean(dim=1)
            aud_context = aud_context.mean(dim=1)
            del context_audio
            aud_outputs = (aud_outputs * self.p + aud_context * (1 - self.p)) / 2
            del aud_context

        if check == "train":
            aud_outputs = self.dropout(aud_outputs)
        aud_outputs = self.linear1(aud_outputs)

        return aud_outputs  # returns [batch_size,output_dim]
