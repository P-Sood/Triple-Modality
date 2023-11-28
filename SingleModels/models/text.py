from transformers import AutoModel , RobertaForSequenceClassification
import torch.nn as nn
import torch
import numpy as np

def collate_batch(batch , must):  # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    text_list = []
    text_mask = []
    label_list = []
    for input, label in batch:
        
        text_list.append(input["input_ids"].tolist()[0])
        text_mask.append(input["attention_mask"].tolist()[0])
        label_list.append(label)
        
    audio_features = {
        "input_ids": torch.Tensor(np.array(text_list)).long(),
        "attention_mask": torch.Tensor(np.array(text_mask)).long(),
        }

    return audio_features, torch.Tensor(np.array(label_list)).long()
class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.dropout = args["dropout"]
        self.output_dim = args["output_dim"]
        self.dataset = args["dataset"]
        self.BertModel = args["BertModel"]
        self.label2id = args["label2id"]
        self.id2label = args["id2label"]
        
        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = 0.75
        
        self.bert = AutoModel.from_pretrained(self.BertModel)

        self.dropout = nn.Dropout(self.dropout)

        self.linear1 = nn.Linear(1024, self.output_dim)


    def forward(self, input_ids, attention_mask, check):
        
        _, text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        del _
        del attention_mask
        del input_ids

        if check == "train":
            text_outputs = self.dropout(text_outputs)

        text_outputs = self.linear1(text_outputs)
        return text_outputs