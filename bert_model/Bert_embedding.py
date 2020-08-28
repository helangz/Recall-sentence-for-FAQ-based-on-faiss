from transformers import BertTokenizer
import torch
import numpy as np
from tqdm import tqdm
import logging


class Bert_embedding(object):
    
    '''
    Paramter
     model: 训练后的bert模型
     device: 'cpu' 或者'gpu',默认 cpu
     dim:  模型的维度,一般为 768
     data_root: 存储模型bert 字典的地址
     max_sentence_length:最大输入长度,默认 32
     log_name: 日志名,默认为 main
    
    
    function:
        convert: 将text 转换为 bert 的格式输入; input: text, output: tokens, mask
        embed_sentence: 编码句子; input: sentence; output: sentence_embedding (768,1)
        embed_sentence_list: input: sens_list: output:matrix 
        sens_to_vector:  返回 {sens:vec} 
        
    example:
        model=torch.load('./bert_model/bert_class3.pkl').bert
        encoder=Bert_embedding(model)
        encoder.embed_sentence()
        
    '''     
    def __init__(self,model,
                 dim=768,
                 data_root='./bert_model/',
                max_sentence_length=32,
                device='cpu',
                log_name_list=['Error_info','Basic_info','User_info'],):
        self.device=device
        self.model=model.to(self.device)
        self.dim=dim
        self.data_root=data_root
        self.max_sentence_length = max_sentence_length
        self.tokenizer = BertTokenizer.from_pretrained(self.data_root)
        
        self.logger_name_list=log_name_list
        self.logger_error=logging.getLogger(f'{self.logger_name_list[0]}.Faiss_search')
        self.logger_basic=logging.getLogger(f'{self.logger_name_list[1]}.Faiss_search')
        self.logger_user=logging.getLogger(f'{self.logger_name_list[2]}.Faiss_search')
        
        self.logger_basic.info(f'create a bert_embedding class, dim is {self.dim}, max_sentence_length is {max_sentence_length },tokenizer from {self.data_root}' )
        
    #将数据转换为格式化输入   
    def convert(self,text):
        
        # 大于长度则截取
        if len(text)>self.max_sentence_length:
            text=text[:self.max_sentence_length]
        tokeniz_text = self.tokenizer.tokenize(text)
        indextokens = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        input_mask = [1] * len(indextokens)
        pad_indextokens = [0]*(self.max_sentence_length-len(indextokens))
        indextokens.extend(pad_indextokens)
        input_mask_pad = [0]*(self.max_sentence_length-len(input_mask))
        input_mask.extend(input_mask_pad)
        #更改为torch 格式
        indextokens = torch.tensor(indextokens,dtype=torch.long)
        input_mask = torch.tensor(input_mask,dtype=torch.long)
        indextokens,input_mask=indextokens.to(self.device),input_mask.to(self.device)
       
        
        return indextokens.unsqueeze(dim=0),input_mask.unsqueeze(dim=0)
    
    
    #编码句子
    def embed_sentence(self,text):
        try:
            indextokens,input_mask=self.convert(text)
            emb=self.model(indextokens,input_mask)[0]
            # 取最后一层的平均,作为编码
            emb=emb.squeeze(0).mean(0)
            if self.device=='cpu':  
                emb=emb.detach().numpy()
            else:
                emb=emb.cpu().detach().numpy()
            return emb 
        except Exception:
            self.logger_error.error("Some thing happended in Embedding text",exc_info = True)
            return False
        
        
    
    #编码句子集
    def embed_sentence_list(self,sentence_list):        
        self.logger_basic.info(f'embedding {len(sentence_list)} sentences')        
        query_matrix=np.zeros((len(sentence_list),self.dim))
        for i,sen in tqdm(enumerate(sentence_list)):    
            query_matrix[i:]=self.embed_sentence(sen)
        return query_matrix
    
    ## 编码返回{sens:vec} 的字典
    def sens_to_vector(self,sens_list):
        sens_dict={}
        for sen in sens_list:    
            sens_dict[sen]=self.embed_sentence(sen)
        return sens_dict