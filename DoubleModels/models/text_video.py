import torch
from torch import nn
from transformers import BertModel , VideoMAEModel , VideoMAEFeatureExtractor , AutoModel
import numpy as np
from numpy.random import choice


def collate_batch(batch , must): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    text_mask = []
    video_list = []
    input_list = []
    label_list = []
    text_list_mask = None
    vid_mask = None
    
    if not must:
        video_context = [torch.empty((1,1,1,1)) ,  torch.empty((1,1,1,1))]
    else:
        video_context = []


    for (input , label) in batch:
        text = input[0]
        input_list.append(text['input_ids'].tolist()[0])
        text_mask.append(text['attention_mask'].tolist()[0])
        vid_features = input[1]#[6:] for debug
        if not must:
            video_list.append(vid_features)
        else:
            video_list.append(vid_features[0])
            video_context.append(vid_features[1])

        
        label_list.append(label)
    batch_size = len(label_list)
    
    text_list_mask = torch.Tensor(np.array(text_mask))
    
    del text_mask
    
    vid_mask = torch.randint(-13, 2, (batch_size, 1568)) # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it, 
    vid_mask[vid_mask > 0] = 0
    vid_mask = vid_mask.bool()
    # now we have a random mask over values so it can generalize better   
    x = torch.count_nonzero(vid_mask.int()).item()
    rem = (1568*batch_size - x)%batch_size
    if rem != 0:
        idx = torch.where(vid_mask.view(-1) == 0)[0]  # get all indicies of 0 in flat tensor
        num_to_change =  rem# as follows from example abow
        idx_to_change = choice(idx, size=num_to_change, replace=False)
        vid_mask.view(-1)[idx_to_change] = 1
    
    text = {'input_ids':torch.Tensor(np.array(input_list)).type(torch.LongTensor) , 
            'attention_mask':text_list_mask,
        }

    visual_embeds = {'visual_embeds':torch.stack(video_list).permute(0,2,1,3,4) , 
            'attention_mask':vid_mask , 
            'visual_context':torch.stack(video_context).permute(0,2,1,3,4) , 
        }
    return [text  , visual_embeds] , torch.Tensor(np.array(label_list))



class BertVideoMAE(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args ,dropout=0.5):
        super(BertVideoMAE, self).__init__()

        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']
        self.dataset = args['dataset']
        
        self.must = True if "must" in str(self.dataset).lower() else False
        self.p = .6

        if self.must:
            self.bert = AutoModel.from_pretrained('jkhan447/sarcasm-detection-RoBerta-base-CR')
        else:
            self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
            
        self.test_ctr = 1
        self.train_ctr = 1
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        self.bert_norm = nn.LayerNorm(768)
        self.vid_norm = nn.LayerNorm(768)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*2, self.output_dim)
        

    
    def forward(self, input_ids , text_attention_mask , video_embeds , video_context , visual_mask  , check = "train"):        
        #Transformer Time
        _, text_outputs = self.bert(input_ids= input_ids, attention_mask=text_attention_mask,return_dict=False)
        del _
        del input_ids
        del text_attention_mask
        text_outputs = self.bert_norm(text_outputs) # 
        vid_outputs = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        del video_embeds
            
        vid_outputs = torch.mean(vid_outputs, dim=1) # Now it has 2 dimensions 
        vid_outputs = self.vid_norm(vid_outputs) 
        
        if self.must:
            vid_context = self.videomae(video_context , visual_mask)[0]
            del video_context
            vid_context = torch.mean(vid_context, dim=1) # Now it has 2 dimensions 
            vid_context = self.vid_norm(vid_context) 
            vid_outputs = (vid_outputs*self.p + vid_context*(1-self.p))/2
            
        del visual_mask
        #Concatenate Outputs
        tav = torch.cat([text_outputs , vid_outputs],dim=1)
        
        del text_outputs
        del vid_outputs

        #Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)

        return tav # returns [batch_size,output_dim]