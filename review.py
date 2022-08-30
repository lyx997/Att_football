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
    #"right_model_path" : "tensorboard/11v11/model_63620352_selfplay.tar",
    #"left_model_path" : "tensorboard/11v11/model_63620352_selfplay.tar",
    #"left_model_path" : "tensorboard/11v11/model_80425728_gat_with_pae.tar",
    #"left_model_path" : "tensorboard\with-pae\model_1500480_att3.tar",
    #"left_model_path" : "tensorboard\with-pae\model_10503360_att2.tar",
    #"left_model_path" : "tensorboard\with-pae\model_49215744_att_def6.tar",
    "left_model_path" : "tensorboard/att/model_3307392.tar",
    "right_model_path" : "tensorboard/att/model_3307392.tar",
    #"left_model_path" : "logs/[06-02]23.10.26_gat_att_def6_latest_reward1/model_41793408.tar",
    #"left_model_path" : "tensorboard\with-pae\model_14704704_selfplay_att_def6_latest.tar",
    #"left_model_path" : "tensorboard\with-pae\model_69022080_att_def6.tar",
    #"right_model_path" : "tensorboard/11v11/model_70522560_pae.tar",
    "k_epoch" : 3,
    
    "arg_max" : True

}

left_fe = FE3()
arg_dict["feature_dims"] = left_fe.get_feature_dims()
left_model = Team12(arg_dict)
#left_model = Gat9(arg_dict)
#left_model = Gat3(arg_dict)
cpu_device = torch.device('cpu')
left_checkpoint = torch.load(arg_dict["left_model_path"], map_location=cpu_device)
left_model.load_state_dict(left_checkpoint['model_state_dict'])
left_hidden = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
         torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))

if args.opp :
    right_fe = FE3()
    right_model = Team12(arg_dict)
    right_checkpoint = torch.load(arg_dict["right_model_path"], map_location=cpu_device)
    right_model.load_state_dict(right_checkpoint['model_state_dict'])
    right_hidden = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
         torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))


def agent(obs, args):
    global left_model
    global left_fe
    global left_hidden
    global steps
    
    steps +=1
    if args.render:
        time.sleep(0.01) 

    obs = obs[0]
    state_dict, opp_num = left_fe.encode(obs, 0.1)
    state_dict_tensor = stt3(state_dict, left_hidden)
    with torch.no_grad():
        a_prob, m_prob, _, left_hidden, attack_att, defence_att = left_model(state_dict_tensor)
        
    if arg_dict["arg_max"]:
        a = torch.argmax(a_prob).item()
    else:
        a = Categorical(a_prob).sample().item()
        
    real_action = 0
    if a==0:
        real_action = int(a)
    elif a==1:
        if arg_dict["arg_max"]:
            m = torch.argmax(m_prob).item()
        else:
            m = Categorical(m_prob).sample().item()
        real_action = int(m + 1)
    else:
        real_action = int(a + 7)

    return [real_action], a, attack_att, defence_att, opp_num

def agent_opp(obs, args):
    global left_model
    global right_model
    global left_fe
    global right_fe
    global left_hidden
    global right_hidden
    global steps
    
    steps +=1
    if args.render:
        time.sleep(0.01) 
    
    left_obs = obs[0]
    right_obs = obs[1]
    
    left_state_dict,_ = left_fe.encode(left_obs,0.1)
    left_state_dict_tensor = stt3(left_state_dict, left_hidden)
    right_state_dict,_ = right_fe.encode(right_obs,0.9)
    right_state_dict_tensor = stt3(right_state_dict, right_hidden)
    with torch.no_grad():
        left_a_prob, left_m_prob, _, left_hidden, _, _ = left_model(left_state_dict_tensor)
        right_a_prob, right_m_prob, _, right_hidden, _, _ = right_model(right_state_dict_tensor)
        
    if arg_dict["arg_max"]:
        left_a = torch.argmax(left_a_prob).item()
        right_a = torch.argmax(right_a_prob).item()
    else:
        left_a = Categorical(left_a_prob).sample().item()
        right_a = Categorical(right_a_prob).sample().item()
        
    left_real_action = 0
    right_real_action = 0
    if left_a==0:
        left_real_action = int(left_a)
    
    elif left_a==1:
        if arg_dict["arg_max"]:
            left_m = torch.argmax(left_m_prob).item()
        else:
            left_m = Categorical(left_m_prob).sample().item()
        left_real_action = int(left_m + 1)
    
    else:
        left_real_action = int(left_a + 7)

    if right_a==0:
        right_real_action = int(right_a)
    elif right_a==1:
        if arg_dict["arg_max"]:
            right_m = torch.argmax(right_m_prob).item()
        else:
            right_m = Categorical(right_m_prob).sample().item()
        right_real_action = int(right_m + 1)
    else:
        right_real_action = int(right_a + 7)
    return [left_real_action, right_real_action]


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