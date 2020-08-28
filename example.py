from log.log_config import logger
from faiss_index import Faiss_Index
from bert_model.Bert_embedding import Bert_embedding
import torch

##初始化索引,默认encoder 为bert.
def init_faiss(bert_path='./bert_model/bert_class3.pkl'):
    model=torch.load(bert_path).bert
    encoder=Bert_embedding(model)
    faiss_index=Faiss_Index(encoder)
    return faiss_index

if __name__='__main__':
    ## 定义索引
    faiss_index=init_faiss()
    
    #### 从问题库中提取问题
    import pandas as pd
    data=pd.read_csv('./test/test.tsv',sep='\t')
    query_all=data.text_a.unique()
    
    ## 建立索引
    faiss_index.Build_index(query_all[:1000])
    faiss_index.save_index()  #保存到本地
    
    ## search
    faiss_index.search_for_sens('你好吗')
    
    ## add 添加问题到索引库
    faiss_index.add('你好吗')
    ## 删除问题
    faiss_index.delete('你好吗')
    
    ## clean
    faiss_index.clean()
    faiss_index.save_index() #覆盖本地的索引
    