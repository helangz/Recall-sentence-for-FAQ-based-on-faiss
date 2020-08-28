import logging
import os
def log_config(name_list,data_root):
    logger_list=[]
    for name in name_list:
        logger = logging.getLogger(name)
        logger.setLevel(level = logging.INFO)
        file_path=f"{data_root}/log_{name}.txt"
        if not os.path.exists(file_path):
            with open(file_path,'w') as f:
                pass
        handler = logging.FileHandler(file_path)
        if name!='Error_info':
            handler.setLevel(logging.INFO)
        else:
            handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger_list.append(logger)
    return logger_list
logger_error,logger_basic,loggger_user= log_config(['Error_info','Basic_info','User_info'],'./log')