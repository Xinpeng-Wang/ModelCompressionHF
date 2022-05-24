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
import logging

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
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()





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

config_inter_1 = {
        "--run-name": "qnli_distill_no_aug_att_kl_inter_4layer_from_Tiny6", 
        "--feature_learn": "att_kl_4from6",
        "--teacher_model": "runs/05_04_2022_23:18",
        "--student_model":" models/students/TinyBERT_General_4L_312D", 
        "--data_dir": "/content/drive/MyDrive/keep/datasets/glue_data/QNLI",
        # "--data_dir": "data_toy/QNLI",
        "--task_name": "qnli",
        # "--output_dir": "models/students/qnli",
        "--max_seq_length": 128,
        "--train_batch_size": 32,
        "--num_train_epochs": 10,
        "--do_lower_case": None,
        "--two_stage_index": 1,
        "--layer_selection": 1234
        }

config_pre_1 = {
        "--run-name": "qnli_distill_no_aug_att_kl_pre_4layer_from_Tiny6",
        "--pred_distill": None, 
        "--teacher_model": "runs/05_04_2022_23:18",
        "--student_model":" runs/05_03_2022_18:22", 
        "--data_dir": "/content/drive/MyDrive/keep/datasets/glue_data/QNLI",
        # "--data_dir": "data_toy/QNLI",
        "--task_name": "qnli",
        # "--output_dir": "models/students/qnli",
        "--max_seq_length": 128,
        "--train_batch_size": 32,
        "--num_train_epochs": 10,
        "--do_lower_case": None,
        "--learning_rate": 3e-5,
        "--eval_step": 100,
        "--two_stage_index": 1

}






config_list_inter = [config_inter_1] #, 'config/task_distill.yaml']


config_list_pre = [config_pre_1] #, 'config/task_distill.yaml']





file_to_restore = {}
# for index, value in enumerate(args_list):
# first stage or all-stage train




if len(config_list_inter) > 0 :
    for config in config_list_inter:
        cmd = 'PYTHONPATH=/content/TinyBERT python task_distill.py ' #% PROJECT_FOLDER

        if type(config) is str:
            config  = yaml.load(Path(config))            

        now = datetime.now()
        current_time = now.strftime("%m_%d_%Y_%R")
        log_path = f"runs/{current_time}"





        config['--output_dir'] = log_path
        config['--run-name'] += '_' + current_time
        # save the feature layer distilled student path for prediction layer distillation
        file_to_restore[config['--two_stage_index']] = log_path



        options = []
        for k, v in config.items():   
            if v is not None:
                options += [ f'{k} {v}']  
            else: options += [ f'{k}']



        cmd += ' '.join(options)


        os.system(cmd)
        
        config_save_path = config['--output_dir'] + '/config.yaml'
        yaml.dump(config, Path(config_save_path))


if len(config_list_pre) > 0 :
    for config in config_list_pre:
        cmd = 'PYTHONPATH=/content/TinyBERT python task_distill.py ' # % PROJECT_FOLDER

        if type(config) is str:
            config  = yaml.load(Path(config)) 
        now = datetime.now()
        current_time = now.strftime("%m_%d_%Y_%R")
        log_path = f"runs/{current_time}"




        config['--output_dir'] = log_path
        config['--run-name'] += '_' + current_time
        if '--two_stage_index' in config.keys():
            config['--student_model'] = file_to_restore[config['--two_stage_index']]

        options = []
        for k, v in config.items():   
            if v is not None:
                options += [ f'{k} {v}']  
            else: options += [ f'{k}']



        cmd += ' '.join(options)

       
        os.system(cmd)
        config_save_path = log_path + '/config.yaml'
        yaml.dump(config, Path(config_save_path))