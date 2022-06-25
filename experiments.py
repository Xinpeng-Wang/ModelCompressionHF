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
from transformer.modeling import TinyBertForPreTraining, BertModel, TinyBertForSequenceClassification

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

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# all_cmds = collections.defaultdict(list)
n_gpu = torch.cuda.device_count()
cur_gpu = 0





# config_path_list_2 =[
#                     # 'config/pre_distill.yaml', 
#                     'config/pre_distill_2.yaml',
#                     # 'config/retrain/inter_pre_2.yaml' 
                    
# ]

specific_config_format = {
        "--run-name": "qnli_distill_no_aug_att_kl_inter_4layer_from_Tiny6", 
        "--feature_learn": "att_kl_4from6",
        "--teacher_model": "runs/05_04_2022_23:18",
        "--student_model":" models/TinyBERT_General_6L_768D", 
        "--data_dir": "/content/drive/MyDrive/keep/datasets/glue_data/QNLI",
        # "--data_dir": "data_toy/QNLI",
        "--task_name": "qnli",
        # "--output_dir": "models/students/qnli",
        "--max_seq_length": 128,
        "--train_batch_size": 32,
        "--num_train_epochs": 10,
        "--do_lower_case": None,
        "--layer_selection": 1234
        }



specific_config_pre_format = {
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

}






# config_list_inter = [config_inter_1] #, 'config/task_distill.yaml']


# config_list_pre = [config_pre_1] #, 'config/task_distill.yaml']





# file_to_restore = {}
# # for index, value in enumerate(args_list):
# # first stage or all-stage train




# if len(config_list_inter) > 0 :
#     for config in config_list_inter:
#         cmd = 'PYTHONPATH=/content/TinyBERT python task_distill.py ' #% PROJECT_FOLDER

#         if type(config) is str:
#             config  = yaml.load(Path(config))            

#         now = datetime.now()
#         current_time = now.strftime("%m_%d_%Y_%R")
#         log_path = f"runs/{current_time}"





#         config['--output_dir'] = log_path
#         config['--run-name'] += '_' + current_time
#         # save the feature layer distilled student path for prediction layer distillation
#         file_to_restore[config['--two_stage_index']] = log_path



#         options = []
#         for k, v in config.items():   
#             if v is not None:
#                 options += [ f'{k} {v}']  
#             else: options += [ f'{k}']



#         cmd += ' '.join(options)


#         os.system(cmd)
        
#         config_save_path = config['--output_dir'] + '/config.yaml'
#         yaml.dump(config, Path(config_save_path))


# if len(config_list_pre) > 0 :
#     for config in config_list_pre:
#         cmd = 'PYTHONPATH=/content/TinyBERT python task_distill.py ' # % PROJECT_FOLDER

#         if type(config) is str:
#             config  = yaml.load(Path(config)) 
#         now = datetime.now()
#         current_time = now.strftime("%m_%d_%Y_%R")
#         log_path = f"runs/{current_time}"




#         config['--output_dir'] = log_path
#         config['--run-name'] += '_' + current_time
#         if '--two_stage_index' in config.keys():
#             config['--student_model'] = file_to_restore[config['--two_stage_index']]

#         options = []
#         for k, v in config.items():   
#             if v is not None:
#                 options += [ f'{k} {v}']  
#             else: options += [ f'{k}']



#         cmd += ' '.join(options)

       
#         os.system(cmd)
#         config_save_path = log_path + '/config.yaml'
#         yaml.dump(config, Path(config_save_path))

task_param = {
    'mnli':{
        'teacher': '',
        'data': '',
    },
    'qnli':{
        'teacher': 'models/teacher_finetuned/qnli',
        'data': '/mounts/data/proj/xinpeng/glue_data/QNLI',
    }
}


def train(config, task_type):
    if task_type == 'task_distill':
        cmd = 'python %s/task_distill.py ' % PROJECT_FOLDER
    elif task_type == 'general_distill':
        cmd = 'python %s/general_distill.py ' % PROJECT_FOLDER
    options = []
    for k, v in config.items():   
        if v is not None:
            options += [ f'{k} {v}']  
        else: options += [ f'{k}']



    cmd += ' '.join(options)

    os.system(cmd)
    log_path = config['--output_dir']
    os.mkdir(log_path)
    config_save_path = log_path + '/config.yaml'
    yaml.dump(config, Path(config_save_path))



def layer_weight_selection(teacher_path, task, selection_list):
    teacher_model = BertModel.from_pretrained(teacher_path)
    state_dict = teacher_model.state_dict()
    teacher_state_dict={}
    for k, v in state_dict.items():
        if 'model.' in k:
            name = k[6 :] # remove `module.`
            teacher_state_dict[name] = v
        elif '_float_tensor' in k:
            continue
        else:
            teacher_state_dict[k] = v



    std_state_dict = {}
    # Embedding
    for w in ['word_embeddings', "position_embeddings"]:
        std_state_dict[f"bert.embeddings.{w}.weight"] = teacher_state_dict[f"embeddings.{w}.weight"]
    for w in ['weight', 'bias']:
        std_state_dict[f"bert.embeddings.LayerNorm.{w}"] = teacher_state_dict[f"embeddings.LayerNorm.{w}"]

    # Transformer Blocks#
    std_idx = 0
    # select teacher layers 
    for teacher_idx in selection_list:
    # for teacher_idx in [0, 7, 11]:
        for layer in ['attention.self.key', 'attention.self.value', 'attention.self.query',
        'attention.output.dense','attention.output.LayerNorm', 'intermediate.dense','output.dense','output.LayerNorm',
        ]:
            for w in ['weight', 'bias']:
                std_state_dict[f"bert.encoder.layer.{std_idx}.{layer}.{w}"] = teacher_state_dict[
                    f"encoder.layer.{teacher_idx}.{layer}.{w}"
                ]
    std_idx += 1

    # LM Head
    for w in ['weight', 'bias']:
        std_state_dict[f"bert.pooler.dense.{w}"] = teacher_state_dict[f"pooler.dense.{w}"]
        
        path = teacher_path + '/' + task + '_' + '_'.join([str(x) for x in selection_list]) + '.pt'
        if not os.path.exists(path):
            torch.save(std_state_dict, path)
    return path



def task_specific_two_stage_training(task, method, student_layer):
    
    ########### intermediate layer training ##############

    # config setting
    config_inter = specific_config_format
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    log_path_inter = f"runs/{current_time}"
    num_epoch = config_inter['--num_train_epochs']


    config_inter['--output_dir'] = log_path_inter
    config_inter['--run-name'] = task + '_' + method + '_' + 'inter' + f'{num_epoch}' + 'epoch' + '_' + f'{student_layer}' + 'S' + '_' + current_time
    config_inter['--feature_learn'] = task
    # config_inter = task_specific_config(config_inter, task, student_layer)
    config_inter['--feature_learn'] = method
    config_inter['--teacher_model'] = task_param[task]['teacher']
    config_inter['--data_dir'] = task_param[task]['data']
    config_inter['--initialize_from'] = 'finetuned_teacher'
    selection_list = [1, 3, 5, 7, 9, 11]
    config_inter['--init_path'] = layer_weight_selection(config_inter['--teacher_model'], task, selection_list)

    # train
    train(config_inter, 'task_distill')


    ########### prediction layer training ###############
    # config setting
    config_pred = specific_config_format
    config_pred['--pred_distill'] = None
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Y_%R")
    log_path_pred = f"runs/{current_time}"
    num_epoch = config_pred['--num_train_epochs']

    config_pred['--student_model'] = log_path_inter
    config_pred['--output_dir'] = log_path_pred
    config_pred['--run-name'] = task + '_' + method + '_' + 'pred' + f'{num_epoch}' + 'epoch' + '_' + f'{student_layer}' + 'S' + '_' + current_time
    config_pred['--feature_learn'] = task
    # config_pred = task_specific_config(config_pred, task, student_layer)
    config_inter['--teacher_model'] = task_param[task]['teacher']
    config_inter['--data_dir'] = task_param[task]['data']


    # train
    train(config_pred, 'task_distill')
    return


task_specific_two_stage_training('qnli', 'attn_kl_val_kl', 6)