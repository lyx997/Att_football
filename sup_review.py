import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time, os
import argparse
import gfootball.env as footballenv
from models.conv1d import Model as PPO
#from models.conv2d import Model as PPO_Conv2
from sup_models.att_off import Model as Offence
from sup_models.att_def import Model as Defence

from sup_encoders.att_encoder import FeatureEncoder as FE, state_to_tensor as stt
#from encoders.encoder_conv2 import FeatureEncoder as FE2, state_to_tensor as stt2
from datetime import datetime, timedelta

if os.path.exists('log.txt'):
    os.remove('log.txt')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--scenario', type=str, default="11_vs_11_competition")
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--opp', type=bool, default=True)
args = parser.parse_args()

arg_dict = {
    "learning_rate" : 0.0002,
    "off_model_path" : "logs/[09-02]23.54.05_gat_conv_seperate_/off/model_off_22807296.tar",
    "def_model_path" : "logs/[09-02]23.54.05_gat_conv_seperate_/def/model_def_23107392.tar",

}

fe_off = FE()
arg_dict["off_feature_dims"] = fe_off.get_feature_dims()
model_off = Offence(arg_dict)
cpu_device = torch.device('cpu')
off_checkpoint = torch.load(arg_dict["off_model_path"], map_location=cpu_device)
model_off.load_state_dict(off_checkpoint['model_state_dict'])

if args.opp :
    fe_def = FE()
    arg_dict["def_feature_dims"] = fe_def.get_feature_dims()
    model_def = Defence(arg_dict)
    def_checkpoint = torch.load(arg_dict["def_model_path"], map_location=cpu_device)
    model_def.load_state_dict(def_checkpoint['model_state_dict'])

def find_most_att_idx(player_att2_idx):
    player_att2_idx = player_att2_idx.squeeze()
    player_most_att_idx = player_att2_idx.sort(descending=True)[1][0]
    return [player_most_att_idx]

if args.opp:
    env = footballenv.create_environment(env_name=args.scenario, logdir='dump', write_full_episode_dumps=True, write_video=True, render=args.render, representation='raw', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=1, other_config_options={'action_set':'v2'})
else:
    env = footballenv.create_environment(env_name=args.scenario, logdir='dump', write_full_episode_dumps=True, write_video=True, render=args.render, representation='raw', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=0, other_config_options={'action_set':'v2'})
obs = env.reset()
    
episode = 0
    
cur_time = datetime.now()    
state_list = ["scrable", "pass", "dribble", "catch", "steal", "stop", "score"] #"scrable"——"争夺", "pass"——"传球", "dribble"——"带球", "catch"——"接球", "steal"——"断球", "stop"——"停球", "score"——"进球"
logfile_path = 'dump/log_'+ cur_time.strftime("[%m-%d]%H.%M.%S") + '.txt'


while True: # episode loop

    done = False
    episode += 1
    off_steps, def_steps, off_acc, def_acc= 0, 0, 0, 0
    env.reset()
    [obs, opp_obs] = env.observation()
    ball_owned_team = obs["ball_owned_team"] #-1
    prev_off_active = obs["active"]
    prev_def_active = opp_obs["active"]
    while not done:
        #attack model
        while not done:  # step loop
            if ball_owned_team == 1: #ball owned by opp change to model_def
                break
           
            state_dict = fe_off.encode(obs)
            state_dict_tensor = stt(state_dict)
            with torch.no_grad():
                left_att_idx, right_att_idx = model_off(state_dict_tensor)
                most_att_idx = find_most_att_idx(left_att_idx)
            
            [obs, opp_obs], [rew, _], done, info = env.att_step([19,19], [most_att_idx,[]])
            #[obs, opp_obs], [rew, _], done, info = env.step([19,19])
            active = obs["active"]
            ball_owned_team = obs["ball_owned_team"] #-1
            state_dict["label_left_att"][:,active] = 1.0
                      
            off_steps += 1
            if active == most_att_idx[0]:
                off_acc += 1
                if prev_off_active != active:
                    print("off model pass acc")
            transition = (state_dict) #change to model defence
            prev_off_active = active
            
            if off_steps % 100 == 0:
                #print("offence model total right", off_acc, "in", off_steps, "accuracy:", off_acc/off_steps)  
                pass
               
        #defence model
        while not done:  # step loop
            if ball_owned_team == 0: #ball owned by opp change to model_def
                break
            is_stopped = False
           
            state_dict = fe_def.encode(obs)
            state_dict_tensor = stt(state_dict)
            with torch.no_grad():
                right_att_idx, left_att_idx = model_def(state_dict_tensor)
                most_att_idx = find_most_att_idx(right_att_idx)
            
            [obs, opp_obs], [rew, _], done, info = env.att_step([19,19], [[],most_att_idx])
            #[obs, opp_obs], [rew, _], done, info = env.step([19,19])
            active = opp_obs["active"]
            ball_owned_team = obs["ball_owned_team"] #-1
            state_dict["label_right_att"][:,active] = 1.0
                        
            def_steps += 1
            if active == most_att_idx[0]:
                def_acc += 1
                if prev_def_active != active:
                    print("def model pass acc")
            transition = (state_dict) #change to model defence
            prev_def_active = active
            
            if def_steps % 100 == 0:
                #print("defence model total right", def_acc, "in", def_steps, "accuracy:", def_acc/def_steps)  
                pass
