# code inherited from 
# https://github.com/intersun/PKD-for-BERT-Model-Compression/blob/master/scripts/run_teacher_prediction.py
import os
import sys
import collections
import torch
from multiprocessing import Pool
import yaml
from ruamel.yaml import YAML
from pathlib import Path
from datetime import datetime
from clearml import Task


# Task.set_credentials(
#      api_host="https://api.community.clear.ml", 
#      web_host="https://app.community.clear.ml", 
#      files_host="https://files.community.clear.ml", 
#      key='UF3ZKMM7JHUM8GECT89V', 
#      secret='n6VbR7TQNANEjafBE46x8u0dAINX4EaykcH0rmP3RjuV6pMWwF'
# )
# def read_config(path):
#     """
#     path: path to config yaml file
#     """
#     with open(path) as f:
#         cfg = yaml.safe_load(f)

#     return cfg
yaml = YAML(typ='rt')
yaml.preserve_quotes = True

def run_process(proc):
    os.system(proc)

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0





# config_path_list_2 =[
#                     # 'config/pre_distill.yaml', 
#                     'config/pre_distill_2.yaml',
#                     # 'config/retrain/inter_pre_2.yaml' 
                    
# ]

config_inter_1 = {"--run-name": "qnli_tiny_bert_noDA", 
        "--teacher_model": "models/teachers-finetuned/qnli",
        "--student_model":" models/students/TinyBERT_General_6L_768D", 
        "--data_dir": "/content/drive/MyDrive/keep/datasets/glue_data/QNLI",
        "--task_name": "qnli",
        # "--output_dir": "models/students/qnli",
        "--max_seq_length": 128,
        "--train_batch_size": 32,
        "--num_train_epochs": 10,
        "--run-name": "qnli_distill_no_aug_tiny",
        "--do_lower_case": None
        }

config_pre_1 = {
        "--pred_distill": None, 
         "--run-name": "qnli_tiny_bert_noDA", 
        "--teacher_model": "models/teachers-finetuned/qnli",
        "--student_model":" models/students/qnli", 
        "--data_dir": "/content/drive/MyDrive/keep/datasets/glue_data/QNLI",
        "--task_name": "qnli",
        # "--output_dir": "models/students/qnli",
        "--max_seq_length": 128,
        "--train_batch_size": 32,
        "--num_train_epochs": 3,
        "--run-name": "qnli_distill_no_aug_tiny",
        "--do_lower_case": None,
        "--learning_rate": 3e-5,
        "--eval_step": 100

}






config_list_1 = [config_pre_1] #, 'config/task_distill.yaml']





file_to_restore = {}
# for index, value in enumerate(args_list):
# first stage or all-stage train




if len(config_list_1) > 0 :
    for config in config_list_1:
        cmd = 'PYTHONPATH=/content/TinyBERT python task_distill.py ' #% PROJECT_FOLDER

        if type(config) is str:
            config  = yaml.load(Path(config))            

        now = datetime.now()
        current_time = now.strftime("%m_%d_%Y_%R")
        log_path = f"runs/{current_time}"





        config['--output_dir'] = log_path
        config['--run-name'] += '_' + current_time

        options = []
        for k, v in config.items():   
            if v is not None:
                options += [ f'{k} {v}']  
            else: options += [ f'{k}']



        cmd += ' '.join(options)


        os.system(cmd)
        
        config_save_path = config['--output_dir'] + '/config.yaml'
        yaml.dump(config, Path(config_save_path))


# if len(config_path_list_2) > 0 :
#     for config_path in config_path_list_2:
#         cmd = 'CUDA_VISIBLE_DEVICES=0 python %s/train.py ' % PROJECT_FOLDER

#         config  = yaml.load(Path(config_path)) 
#         DATA_PATH = ' ' + config['data_path'] + ' '

#         now = datetime.now()
#         current_time = now.strftime("%m_%d_%Y_%R")
#         log_path = f"checkpoints/runs/{current_time}"




#         config['args']['--save-dir'] = log_path
#         config['args']['--tensorboard-logdir'] = log_path
#         config['args']['--run-name'] += '_' + current_time

#         cmd += DATA_PATH
#         options = []
#         for k, v in config['args'].items():   
#             if v is not None:
#                 options += [ f'{k} {v}']  
#             else: options += [ f'{k}']



#         cmd += ' '.join(options)

#         os.mkdir(log_path)
#         config_save_path = log_path + '/config.yaml'
#         yaml.dump(config, Path(config_save_path))
#         os.system(cmd)