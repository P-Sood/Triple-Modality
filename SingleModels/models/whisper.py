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
    context_timings_list = []
    #
    for input, label in batch:
        if not must:
            speech_list.append(FEAT(input[1] , sampling_rate=16000).input_features[0])
        else:
            
            context_timings_list.append(input[-1][0]) # The 0 is always context
            speech_list.append(FEAT(input[1], sampling_rate=16000).input_features[0])
            speech_list_context.append(FEAT(input[2], sampling_rate=16000).input_features[0])

        label_list.append(label)

    if must:
        speech_list_context_input_values = torch.Tensor(np.array(speech_list_context))

    audio_features = {
        "audio_features": torch.Tensor(np.array(speech_list)),
        "context_audio": speech_list_context_input_values,
        "context_timings_list": context_timings_list,
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
        self.p = 0.75

        self.whisper = WhisperForAudioClassification.from_pretrained("openai/whisper-medium")

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

    def forward(self, audio_features, context_audio , context_timings_list, check):
        aud_outputs = self.whisper.encoder(audio_features)[0][:,:512,:]
        aud_outputs = aud_outputs.mean(dim=1)
        del audio_features
        if self.must:
            aud_context = self.whisper.encoder(context_audio)[0]
            del context_audio
            new_aud_context = torch.zeros_like(aud_context[:,:512,:]) # Cut it to be this, now assign it
            for i , row in enumerate(aud_context):
                if context_timings_list[i] == None:
                    new_aud_context[i] = row[:512] # If less then 10.24 seconds then take first 10.24 seconds
                    
                elif context_timings_list[i][1] - context_timings_list[i][0] < 10.24:
                    new_aud_context[i] = row[:512] # If less then 10.24 seconds then take first 10.24 seconds
                    
                else: # Take the last 10.24 seconds
                    new_aud_context[i] = row[-512:]
            del aud_context
                
            new_aud_context = new_aud_context.mean(dim=1)
            aud_outputs = (aud_outputs * self.p + new_aud_context * (1 - self.p)) / 2
            del new_aud_context

        if check == "train":
            aud_outputs = self.dropout(aud_outputs)
        aud_outputs = self.linear1(aud_outputs)

        return aud_outputs  # returns [batch_size,output_dim]
