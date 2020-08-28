# 目录
1. 项目介绍
2. 使用方法介绍
3. 基本参数说明
4. 注意事项
5. 未来的改进方向

## 项目介绍
问答召回系统; 输入问题,返回最相似的Topk个问题.

    例: 输入: 你好吗?
        返回:[how are you, 你今天怎样?]
## 使用方法介绍
步骤1. 定义encoder.

    from log.log_config import logger_error,logger_basic,loggger_user
    from bert_model.Bert_embedding import Bert_embedding
    import torch
    # bert 编码
    model=torch.load(bert_path).bert
    encoder=Bert_embedding(model)

步骤2. 读取索引

    faiss_index=Faiss_Index(encoder)
    # 如果已经建立了索引，则下面两行代码可省略
    faiss_index.Build_index(sens_list)
    faiss_index.save_index()
步骤3. 查找

    faiss_index.search_for_sens(text,n=5)
其他功能
  
* 在索引里面添加句子

    faiss_index.add(sentence_list)
* 删除句子

    faiss_index.delete(sentence_list)
* 替换句子

    faiss_index.update(old_sentence,new_sentence)
    
* 清除索引（初始化）

    faiss_index.clean()


## 模型的基本参数说明
### Bert_embedding

    Paramter
     model: 预训练后的bert模型
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

### faiss_index
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
        
        
    

### requirement
主要的环境依赖，完整的参见requirement.txt文件。

    faiss-cpu==1.6.3
    torch==1.1.0
    torchvision==0.2.2
    transformers==3.0.2
### 文件夹说明
1. bert_model: 存放bert模型及其加载模型的相关配置,和bert_embedding 方法
2. index_file: 存放索引的文件夹,保存已经建立的索引
3. log 日志文件和日志配置的存放地方,其中日志文件分为,记录用户输入和输出的user_info,记录管理员操作的basic info,记录系统错误的error_info.
4. model_graph 加载bert 模型的配套工具
5. test 存放测试用例
6. faiss_index.py 索引工具


## 注意事项
1. 在使用索引搜索前，一定要注意已经建立了索引.
2. 建立索引的时候，输入的句子集合一定长度要大于默认值100
3. 使用build_index建立新索引的时候，重新初始化索引。
4. 调用 save_index()的时候，会覆盖原有的文件，因此需要提前备份。
5. 在运行模型的时候，要确保每个文件夹都存在。
6. 在模型使用的时候,要载入日志

## 未来的改进方向
* 可以使用多种方式构建索引，现在只提供了句子的方式，未来将会提供数组和字典等方式。

* 提供索引合并的功能。

* 现在只提供了基于bert模型的编码方法，未来将提供更多编码方法的接口。

* 将pytorch改为tensorflow, tensorflow cpu 比pytorch 运行速度更快。

* 提供多进程的方法加速离线编码方法。 
