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
from models.gat_att3 import Model as Gat3
from models.gat_att2 import Model as Gat2
from models.gat_att_def6_latest11 import Model as Gat6
from models.gat_att_def10 import Model as Gat10
from models.team_opp_attention10 import Model as Team10
from models.team_opp_attention25 import Model as Team12
from models.opp_attention import Model as Opp
from encoders.encoder_basic import FeatureEncoder as FE, state_to_tensor as stt
#from encoders.encoder_conv2 import FeatureEncoder as FE2, state_to_tensor as stt2
from encoders.encoder_gat_att_def_latest12 import FeatureEncoder as FE3, state_to_tensor as stt3
from datetime import datetime, timedelta

if os.path.exists('log.txt'):
    os.remove('log.txt')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--scenario', type=str, default="11_vs_11_stochastic")
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--opp', type=bool, default=True)
args = parser.parse_args()

arg_dict = {
    "lstm_size" : 256,
    "learning_rate" : 0.0002,
    "gamma" : 0.992,
    "lmbda" : 0.96,
    "entropy_coef" : 0.0,
    "move_entropy_coef" : 0.0,
    "k_epoch" : 3,
    
    "arg_max" : True

}

if args.opp:
    env = footballenv.create_environment(env_name=args.scenario, logdir='dump', write_full_episode_dumps=True, write_video=True, render=args.render, representation='raw', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=1, other_config_options={'action_set':'v2'})
else:
    env = footballenv.create_environment(env_name=args.scenario, logdir='dump', write_full_episode_dumps=True, write_video=True, render=args.render, representation='raw', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=0, other_config_options={'action_set':'v2'})
obs = env.reset()
    
episode = 0
    
def split_att_def_idx_(attack_att, defence_att, active_idx):
    attack_right_idx1 = attack_att[0].sort(descending=True)[1]
    attack_right_idx2 = attack_att[1].sort(descending=True)[1]
    attack_left_idx = attack_att[2].sort(descending=True)[1]
    defence_left_idx1 = defence_att[0].sort(descending=True)[1]
    defence_left_idx2 = defence_att[1].sort(descending=True)[1]
    defence_right_idx = defence_att[2].sort(descending=True)[1]
    
    for i, idx in enumerate(attack_left_idx):
        if idx >= active_idx[0]:
            attack_left_idx[i] = idx + 1
    for i, idx in enumerate(defence_left_idx1):
        if idx >= active_idx[0]:
            defence_left_idx1[i] = idx + 1
    for i, idx in enumerate(defence_left_idx2):
        if idx >= active_idx[0]:
            defence_left_idx2[i] = idx + 1

    team_att_idx = [attack_left_idx, defence_left_idx1, defence_left_idx2]
    opp_att_idx = [attack_right_idx1, attack_right_idx2, defence_right_idx]

    return team_att_idx, opp_att_idx

def split_def_idx(defence_att, active_idx):
    defence_left_idx1 = defence_att[0].sort(descending=True)[1][0]
    defence_left_idx2 = defence_att[1].sort(descending=True)[1][0]
    defence_right_idx = defence_att[2].sort(descending=True)[1][0]
    
    if defence_left_idx1 >= active_idx[0]:
        defence_left_idx1 += 1
    if defence_left_idx2 >= active_idx[0]:
        defence_left_idx2 += 1

    team_att_idx = [int(defence_left_idx1), int(defence_left_idx2)]
    opp_att_idx = [int(defence_right_idx)]

    return team_att_idx, opp_att_idx

def split_att_def_idx(attack_att, defence_att, active_idx, opp_idx):
    #attack_right_idx1 = attack_att[1].squeeze().sort(descending=True)[1][0]
    #attack_right_idx2 = attack_att[1].sort(descending=True)[1][0]
    attack_left_idx = attack_att[0].squeeze().sort(descending=True)[1][0]
    #defence_left_idx1 = defence_att[1].squeeze().sort(descending=True)[1][0]
    #defence_left_idx2 = defence_att[1].sort(descending=True)[1][0]
    defence_right_idx = defence_att[0].squeeze().sort(descending=True)[1][0]
    
    if attack_left_idx >= active_idx[0]:
        attack_left_idx += 1
    #if defence_left_idx1 >= active_idx[0]:
    #    defence_left_idx1 += 1
    #if attack_right_idx >= opp_idx:
    #    attack_right_idx += 1
    if defence_right_idx >= opp_idx:
        defence_right_idx += 1
    #if defence_left_idx2 >= active_idx[0]:
    #    defence_left_idx2 += 1

    #team_att_idx = [int(attack_left_idx), int(defence_left_idx1), int(defence_left_idx2)]
    team_att_idx = [int(attack_left_idx)]
    #opp_att_idx = [int(attack_right_idx1), int(attack_right_idx2), int(defence_right_idx)]
    opp_att_idx = [int(defence_right_idx)]

    #att_entropy = - torch.mean(torch.log(attack_att[2]))
    #print("entropy", att_entropy)

    return team_att_idx, opp_att_idx
    
    
cur_time = datetime.now()    
state_list = ["scrable", "pass", "dribble", "catch", "steal", "stop", "score", "fly"] #"scrable"——"争夺", "pass"——"传球", "dribble"——"带球", "catch"——"接球", "steal"——"断球", "stop"——"停球", "score"——"进球"
logfile_path = 'dump/log_'+ cur_time.strftime("[%m-%d]%H.%M.%S") + '.txt'

while True:
    steps = 0
    episode += 1
    done = 0
    init_left_score = 0
    obs = env.reset()
    active = [obs[0]["active"], obs[1]["active"]]
    state_dict = {state_list[0]: active}
    owned_ball_team = None
    owned_ball_player = None
    prev_obs = {}
    #prev_active = []
    prev_orient = None
    prev_owned_ball_team = None
    prev_owned_ball_player = None
    pass_or_not = False
    with open(logfile_path, "a") as f:
        f.writelines("###################################################")
        f.writelines('\n')
        while not done:
            
            if owned_ball_team != -1:
                prev_owned_ball_team = owned_ball_team
                prev_owned_ball_player = owned_ball_player
            #elif prev_obs:
            #    prev_owned_ball_team = prev_obs[0][0]["ball_owned_team"]
            #    prev_owned_ball_player = prev_obs[0][0]["ball_owned_player"]
                #prev_active = [prev_obs[0]["active"], prev_obs[1]["active"]]
            #if args.opp:
            #    action = agent_opp(obs, args)
            #else:
            #    action, a, attack_att, defence_att, opp_num = agent(obs, args)
            #    active_idx = [obs[0]["active"]]
            #    team_att_idx, opp_att_idx = split_att_def_idx(attack_att, defence_att, active_idx, opp_num)
            #    #team_att_idx, opp_att_idx = split_att_def_idx_(attack_att, defence_att, active_idx)

            obs = env.att_step([19, 19], [[],[]])
            active = [obs[0][0]["active"], obs[0][1]["active"]]
            owned_ball_team = obs[0][0]["ball_owned_team"]
            owned_ball_player = obs[0][0]["ball_owned_player"]

            left_active_x, left_active_y = obs[0][0]["left_team"][active[0]]
            right_active_x, right_active_y = obs[0][0]["right_team"][active[1]]
            ball_x, ball_y, ball_z = obs[0][0]["ball"]
            dis_left_to_ball = np.linalg.norm([ball_x-left_active_x, ball_y-left_active_y])
            dis_right_to_ball = np.linalg.norm([ball_x-right_active_x, ball_y-right_active_y])
            dis_to_ball = [dis_left_to_ball, dis_right_to_ball]

            if dis_to_ball[0] < 0.02 and dis_to_ball[0] < dis_to_ball[1]:
                owned_ball_team = 0
                owned_ball_player = active[0]
            elif dis_to_ball[1] < 0.02 and dis_to_ball[1] < dis_to_ball[0]:
                owned_ball_team = 1
                owned_ball_player = active[1]
            
            if owned_ball_team != -1:
                prev_obs = obs
            #obs = env.att_step(action, [[],[]])
            left_score = obs[1][0]

            if left_score != 0:
                state_dict = {state_list[6]:{prev_owned_ball_team: prev_owned_ball_player}}
                prev_obs = {}
                #prev_active = []
                prev_owned_ball_team = None
                prev_owned_ball_player = None
                pass_or_not = False

            elif not prev_obs:
                state_dict = {state_list[0]: active} #争球
            elif prev_owned_ball_team == owned_ball_team and prev_owned_ball_player == owned_ball_player:
                state_dict = {state_list[2]: {owned_ball_team: owned_ball_player}} #带球
                pass_or_not = False
            elif prev_owned_ball_team != owned_ball_team and owned_ball_team != -1 and prev_owned_ball_team != None:
                state_dict = {state_list[4]: [{owned_ball_team: owned_ball_player}, {prev_owned_ball_team: prev_owned_ball_player}]} #断球
                pass_or_not = False
            elif prev_owned_ball_team == owned_ball_team or (prev_owned_ball_team == None and owned_ball_team != -1):
                state_dict = {state_list[3]: [{owned_ball_team: owned_ball_player}, {prev_owned_ball_team: prev_owned_ball_player}]} #接传球
                pass_or_not = False
            elif prev_owned_ball_player == active[prev_owned_ball_team]:
                state_dict = {state_list[5]: [{prev_owned_ball_team: prev_owned_ball_player}]} #停球
                pass_or_not = False
            else:
                if not pass_or_not:
                    state_dict = {state_list[1]: [{prev_owned_ball_team: prev_owned_ball_player}, {prev_owned_ball_team: active[prev_owned_ball_team]}]} #传球
                    pass_or_not = True
                else:
                    state_dict = {state_list[7]: [{prev_owned_ball_team: prev_owned_ball_player}, {prev_owned_ball_team: active[prev_owned_ball_team]}]} #球飞在空中
            

            left_score += init_left_score
            init_left_score = left_score

            done = obs[2]
            obs = obs[0]
            steps += 1
            print("episode:", episode, "step:", steps, "score:", left_score, "state:", state_dict)

            f.writelines(str(["episode:", episode, "step:", steps, "score:", left_score, "state:", state_dict])) 
            f.writelines('\n')
        f.writelines('\n')