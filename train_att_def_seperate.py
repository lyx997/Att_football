import gfootball.env as football_env
import time, pprint, json, os, importlib, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

from actor import *
from learner import *
#from evaluator_with_hard import evaluator
from evaluator import evaluator
from datetime import datetime, timedelta


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"]+"/args.json","w")
    f.write(args_info)
    f.close()

def copy_models(dir_src, dir_dst): # src: source, dst: destination
    # retireve list of models
    l_cands = [f for f in os.listdir(dir_src) if os.path.isfile(os.path.join(dir_src, f)) and 'model_' in f]
    l_cands = sorted(l_cands, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"models to be copied: {l_cands}")
    for m in l_cands:
        shutil.copyfile(os.path.join(dir_src, m), os.path.join(dir_dst, m))
    print(f"{len(l_cands)} models copied in the given directory")
    
def main(arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cur_time = datetime.now()
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    save_args(arg_dict)
    if arg_dict["trained_model_path"] and 'kaggle' in arg_dict['env']: 
        copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir'])

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)

    fe_att_def = importlib.import_module("encoders." + arg_dict["encoder"])
    fe_att_def = fe_att_def.FeatureEncoder()
    arg_dict["feature_dims"] = fe_att_def.get_feature_dims()

    model_att = importlib.import_module("models.gat_att")
    model_def = importlib.import_module("models.gat_def")
    cpu_device = torch.device('cpu')

    center_model_att = model_att.Model(arg_dict)
    center_model_def = model_def.Model(arg_dict)
    
    att_optimization_step = 0
    def_optimization_step = 0

    model_att_dict = {
        'optimization_step': att_optimization_step,
        'model_state_dict': center_model_att.state_dict(),
        'optimizer_state_dict': center_model_att.optimizer.state_dict(),
    }
    model_def_dict = {
        'optimization_step': def_optimization_step,
        'model_state_dict': center_model_def.state_dict(),
        'optimizer_state_dict': center_model_def.optimizer.state_dict(),
    }
    
    att_path = arg_dict["log_dir"]+f"/model_att_{att_optimization_step}.tar"
    torch.save(model_att_dict, att_path)
    def_path = arg_dict["log_dir"]+f"/model_def_{def_optimization_step}.tar"
    torch.save(model_def_dict, def_path)

        
    center_model_att.share_memory()
    center_model_def.share_memory()

    center_model = [center_model_att, center_model_def]

    att_data_queue = mp.Queue()
    att_signal_queue = mp.Queue()
    def_data_queue = mp.Queue()
    def_signal_queue = mp.Queue()

    data_queue = [att_data_queue, def_data_queue]
    signal_queue = [att_signal_queue, def_signal_queue]

    summary_queue = mp.Queue()
    
    processes = [] 
    for i in range(2):
        p = mp.Process(target=learner, args=(center_model[i], data_queue[i], signal_queue[i], summary_queue, arg_dict))
        p.start()
        processes.append(p)
    for rank in range(arg_dict["num_processes"]):
        if arg_dict["env"] == "11_vs_11_selfplay":
            p = mp.Process(target=actor_self, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        else:
            p = mp.Process(target=actor, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)
    for i in range(5):
        if "env_evaluation" in arg_dict:
            p = mp.Process(target=evaluator, args=(center_model, signal_queue, summary_queue, arg_dict))
            p.start()
            processes.append(p)
        
    for p in processes:
        p.join()
    

if __name__ == '__main__':

    arg_dict = {
        "env": "11_vs_11_competition",    
        # "11_vs_11_selfplay" : environment used for self-play training
        # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
        # "11_vs_11_kaggle" : environment used for training against fixed opponent(rule-based AI hard)
        "num_processes": 30,  # should be less than the number of cpu cores in your workstation.
        "batch_size": 12,   
        "buffer_size": 6,
        "rollout_len": 30,

        "lstm_size" : 256,
        "k_epoch" : 3,
        "learning_rate" : 0.0001,
        "gamma" : 0.993,
        "lmbda" : 0.96,
        "entropy_coef" : 0.0001,
        "grad_clip" : 3.0,
        "eps_clip" : 0.1,

        "summary_game_window" : 10, 
        "model_save_interval" : 300000,  # number of gradient updates bewteen saving model

        "trained_model_path" : 'logs/[04-01]22.16.37/model_79525440.tar', # use when you want to continue traning from given model.
        "latest_ratio" : 0.5, # works only for self_play training. 
        "latest_n_model" : 10, # works only for self_play training. 
        "print_mode" : False,

        "encoder" : "encoder_gat3",
        "rewarder" : "rewarder_basic",
        #"model" : "gat_att_def3",#add left right closest
        "algorithm" : "ppo",

        "env_evaluation":'logs/selfplay/model_63620352_selfplay.tar'  # for evaluation of self-play trained agent (like validation set in Supervised Learning)
    }
    
    main(arg_dict)