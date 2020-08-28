import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
import pandas as pd
data_root='./bert_model/bert_wwm2/'
class BertclassModel(nn.Module):
    def __init__(self,num_classes):
        super(BertclassModel,self).__init__()
        self.num_classes=num_classes
 
        self.bert = BertModel.from_pretrained(data_root).cuda()
        for param in self.bert.parameters():
            param.requires_grad = True 
        self.fc = nn.Linear(768, self.num_classes)       
    def forward(self, indextokens,input_mask):
        embedding=self.bert(indextokens,input_mask)[0]
        embedding=torch.mean(embedding,1)
        out=self.fc(embedding)
        return out