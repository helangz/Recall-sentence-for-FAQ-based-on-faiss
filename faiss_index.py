import faiss
from annoy import AnnoyIndex
import random  
from tqdm import tqdm
import numpy as np
import json
import os
import logging
class  Faiss_Index(object):
    '''
    Paramter
     encoder: 编码方法,主要包括embed_sentence和embed_sentence_list 两种方法
     dim: 维度 默认为768
     nlist:  faiss中的聚类中心数目,默认为100
     nprobe: faiss中查找的聚类中心的数目,,默认为10
     index_path: 存放faiss索引的位置,默认为 './index_file/faiss.index'
     sens_id_path: 存放setences-id 的位置,默认为 './index_file/sens_id.json'
     id_sens_path: 存放id-sentences的位置,默认为:'./index_file/id_sens.json'
    
    
    function:
        Build_index: 借助faiss建立问题索引库
        add: 添加新的问题
        delete:删除问题库的问题
        update: 替换问题
        
        search: 返回问题库中最相似问题的索引
        search_for_sens: 返回问题库中与该问题最相似的TOPk个句子,默认返回5个
        search_for_sens_list: 返回列表
        save_index:  存储问题索引
        clean: 重新初始化索引
        
    example:
        
        faiss_index=Faiss_Index(encoder)
        ## 新建索引
        faiss_index.Build_index(sens_list)
        faiss_index.save()
        ## 搜索
        faiss_index.search(sens)
        
    '''  
    def __init__(self,
                 encoder,
                 dim=768,
                 nlist=100,
                 nprobe=10,
                 index_path='./index_file/faiss.index',
                 sens_id_path='./index_file/sens_id.json',
                 id_sens_path='./index_file/id_sens.json',
                 log_name_list=['Error_info','Basic_info','User_info'],
                ): 
        self.encoder=encoder  
        self.dim=dim
        self.nlist=nlist
        self.nprobe=nprobe
        self.index_path=index_path
        self.sens_id_path=sens_id_path
        self.id_sens_path=id_sens_path
        self.logger_naem_list=log_name_list
        self.logger_error=logging.getLogger(f'{self.logger_naem_list[0]}.Faiss_search')
        self.logger_basic=logging.getLogger(f'{self.logger_naem_list[1]}.Faiss_search')
        self.logger_user=logging.getLogger(f'{self.logger_naem_list[2]}.Faiss_search')
        
        
        self.index=None
        ##faiss_index.index.ntotal

        if os.path.exists(self.index_path) and os.path.exists(self.sens_id_path) and os.path.exists(self.id_sens_path):
            self.logger_basic.info('load file_path')
            self.index=faiss.read_index(index_path)
            self.logger_basic.info('load sentence id file')
            with open(self.sens_id_path,'r')  as f:
                self.sens_id=json.load(f)
            self.logger_basic.info('load sentence id file')
            with open(self.id_sens_path,'r')  as f:
                self.id_sens=json.load(f)
        else:
            self._init_index()
            self.logger_basic.warning("You need build a new index")


    
    ## 初始化树
    def _init_index(self):
        quantizer = faiss.IndexFlatIP(self.dim)  # 定义量化器
        ## 如果是L2则需要 对索引映射  faiss.IndexIDMap
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2) #也可采用向量內积               
        self.index.nprobe = self.nprobe
        self.logger_basic.info(f"you create a new index, dim is {self.dim},nlist is {self.nlist}, nprobe is {self.nprobe}")

    ## 构建索引
    def Build_index(self,sens_list): 
        ## 判断输入的格式 
        
        #异常情况1 sens_list 是字符串格式
        if type(sens_list)==str:
            self.logger_error.error("input error, you should input a list")
            return 'type error'
        ## 异常情况2 sens_list 不是列表等可迭代类型.
        try:
            sens_list=list(set(sens_list))
            ids_list=list(range(len(sens_list)))
        except:
            self.logger_error.error("input error,you shoule  input a sentences list")
            return "input error"
        
        ## 异常情况3, 长度小于100
        if len(sens_list)<100:
            self.logger_error.error("input error,you shoule input a unique sentence list longer than 100.")
            return "lenght error"
        ## 异常情况4,列表里面的元素不是字符串
        for i in sens_list:
            if type(i)!=str:
                self.logger_error.error("input error,sens_list contain some element is not str.")
                return "input error"
        
        
        try:
            self._init_index()
            self.logger_basic.info(f"starting embedding {len(ids_list)} questions...")
            query_matrix=self.encoder.embed_sentence_list(sens_list)
            self.index.train(query_matrix.astype('float32'))
            self.index.add_with_ids(query_matrix.astype('float32'),np.array(ids_list))
            # 保存sentence_id的索引
            self.sens_id={sen:ids for ids,sen in zip(ids_list,sens_list)}
            self.id_sens={str(ids):sen for ids,sen in zip(ids_list,sens_list)}
        except Exception:
            self.logger_error.error("Build_matrix error",exc_info = True)
            return "Build_matrix error"
        
    ## 添加
    def add(self,sens_list):
        
        if type(sens_list)==str:
            sens_list=[sens_list]
        ## 异常情况1 sens_list 不是列表等可迭代类型.
        try:
            sens_list=list(set(sens_list))
        except:
            self.logger_error.error("input error,you shoule  input a sentences list")
            return "input error"
        
        ## 异常情况2 ,列表里面的元素不是字符串
        for i in sens_list:
            if type(i)!=str:
                self.logger_error.error("input error,sens_list contain some element is not str.")
                return "input error"

            
        
        #需要添加的字
        add_num=len(sens_list)
        try:
            add_num=len(sens_list) ## 输出的长度
            sens_list=list(set(sens_list))
            
            ## 找到现在的sens_id里面的最后一个值,在它后面添加.
            if len(self.id_sens)<1:
                last_id=1
            else:
                last_ids=int([key for key in self.id_sens][-1])
            ## 查找需要添加的元素在不在索引里面,如果不在则添加
                    ## 新增句子
            new_add=[] #新增句子
            id_list=[] #新增句子的索引
            i=0       ##新增数量
            for sen in sens_list:
                if self.sens_id.get(sen) is not None:
                    pass
                else:
                    i=i+1
                    self.sens_id[sen]=last_ids+i
                    self.id_sens[str(last_ids+i)]=sen
                    new_add.append(sen)
                    id_list.append(last_ids+i)
        
            ## 添加元素
            query_matrix=self.encoder.embed_sentence_list(new_add)
            id_list=np.array(id_list)
            self.index.add_with_ids(query_matrix.astype('float32'),id_list)   
            self.logger_basic.info(f'add {i} sentence, {add_num-i} is already in index')
        except:
            
            self.logger_error.error('add some thing wrong',exc_info = True)
            return 'add error'
            
            

    ## 删除
    def delete(self,sens_list):
        ## 输入可以是 列表和字符串
        if type(sens_list)==str:
            sens_list=[sens_list]
        ## 异常情况1 sens_list 不是列表等可迭代类型.
        try:
            sens_list=list(set(sens_list))
        except:
            self.logger_error.error("input error,you shoule  input a sentences list")
            return "input error"
        
        ## 异常情况2 ,列表里面的元素不是字符串
        for i in sens_list:
            if type(i)!=str:
                self.logger_error.error("input error,sens_list contain some element is not str.")
                return "input error"    

        sens_list=list(set(sens_list))
        ids=[int(self.sens_id.get(sen)) for sen in sens_list if self.sens_id.get(sen)]
        if ids:
            self.index.remove_ids(np.array(ids))
        self.logger_basic.info(f'deleted {len(ids)} sentence in index, failed to find {len(sens_list)-len(ids)}')

    ## 更新
    def update(self,sens_list1,sens_list2):
        self.delete(sens_list1)
        self.add(sens_list2)
        self.logger_basic.info(f'add {len(sens_list1)} sentence  and delete {len(sens_list2)} sentence')
    
    ## 查找
    def search(self,text,n=5):
       ## 判断用户输入 是否符合要求.        
       ## 输入进行查找 
        _,index=self.index.search(np.expand_dims(self.encoder.embed_sentence(text),0).astype('float32'), n) 
        index=index[0]
        return index 

    
    ## 查找sentences
    def search_for_sens(self,text,n=5):
        
        ## 异常情况1 没有建立索引
        if not self.index.is_trained:   
            self.logger_error.error('before searching a sentence, you need to build an index ')
            return 'index error'
        ## 异常情况2, 输入text 不是str格式
        if type(text)!=str:   
            self.logger_error.error('error input type, your should input a string')
            return 'input error'
        ## 异常情况3, 输入text 为空
        if len(text)<1:   
            self.logger_error.error('get None input')
            return 'input error'
        ## 异常情况4, 输入text 过长,
        if len(text)>50:   
            self.logger_error.error('get a longger input more than 50')
            return 'input error'
        
        
        
        try:
            ids=self.search(text,n)  
            sentences=[self.id_sens.get(str(i)) for i in ids if self.id_sens.get(str(i))]
            self.logger_user.info(f'search for {text}, and return {str(sentences)}  sentences')
            return sentences
        except Exception:
            self.logger_error.error("Some thing happended in finding text",  exc_info = True)
            return 'embedding error'
    
    
    ## 查找多个sentences
    def search_for_sens_list(self,sens_list,n=5):     
        return [self.search(text) for text in sens_list]
    
    ## 保存index
    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.sens_id_path,'w') as f:
                json.dump(self.sens_id,f)
            with open(self.id_sens_path,'w') as f:
                json.dump(self.id_sens,f)
            self.logger_basic.info(f'save your index in {self.index_path}, and save your sentence-id in {self.sens_id_path},id-sentence in {self.sens_id_path}')
        except Exception:
            self.logger_error.error("save failed,please check the path is exists",  exc_info = True)
            return 'save error'
            
    #重新初始化索引
    def clean(self):
        self._init_index()
        self.sens_id={}
        self.id_sens={}
        self.logger_basic.info('Cleaned index')