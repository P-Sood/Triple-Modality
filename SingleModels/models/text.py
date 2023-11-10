from transformers import AutoModel
import torch.nn as nn
import torch

class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.dropout = args["dropout"]
        self.output_dim = args["output_dim"]
        self.dataset = args["dataset"]
        self.BertModel = args["BertModel"]
        self.hidden_size = args["hidden_size"]
        
        self.bert = AutoModel.from_pretrained(self.BertModel)

        self.dropout = nn.Dropout(self.dropout)

        self.linear1 = nn.Linear(self.bert.encoder.layer[0].output.dense.out_features, self.output_dim)


    def forward(self, input_ids, attention_mask, check):
        _, text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        del _
        del attention_mask
        del input_ids
        torch.cuda.empty_cache()

        if check == "train":
            text_outputs = self.dropout(text_outputs)

        text_outputs = self.linear1(text_outputs)
        return text_outputs