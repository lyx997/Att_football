import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time, os
import argparse
import gfootball.env as footballenv
from sup_models.att_off2 import Model as Offence
from sup_models.att_def2 import Model as Defence

from sup_encoders.att_encoder import FeatureEncoder as FE, state_to_tensor as stt
from datetime import datetime, timedelta

if os.path.exists('log.txt'):
    os.remove('log.txt')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--scenario', type=str, default="11_vs_11_stochastic")
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--opp', type=bool, default=True)
args = parser.parse_args()

arg_dict = {
    "learning_rate" : 0.0002,
    "off_model_path" : "logs/[09-06]21.42.41_gat_conv_seperate_/off/model_off_3102976.tar",
    "def_model_path" : "logs/[09-06]21.42.41_gat_conv_seperate_/def/model_def_3102976.tar",

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
    return {int(player_most_att_idx): float(player_att2_idx[player_most_att_idx])}

if args.opp:
    env = footballenv.create_environment(env_name=args.scenario, logdir='dump', write_full_episode_dumps=True, write_video=True, render=args.render, representation='raw', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=1, other_config_options={'action_set':'v2'})
else:
    env = footballenv.create_environment(env_name=args.scenario, logdir='dump', write_full_episode_dumps=True, write_video=True, render=args.render, representation='raw', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=0, other_config_options={'action_set':'v2'})
obs = env.reset()
    
episode = 0
    
cur_time = datetime.now()    
state_list = ["scrable", "pass", "dribble", "catch", "steal", "stop", "score", "fly"] #"scrable"——"争夺", "pass"——"传球", "dribble"——"带球", "catch"——"接球", "steal"——"断球", "stop"——"停球", "score"——"进球"
logfile_path = 'dump/log_'+ cur_time.strftime("[%m-%d]%H.%M.%S") + '.txt'

while True:
    steps = 0
    episode += 1
    done = 0
    init_left_score = 0
    [obs, opp_obs] = env.reset()
    active = [obs["active"], opp_obs["active"]]
    state_dict = {state_list[0]: active}
    ball_owned_team = None
    ball_owned_player = None
    prev_obs = {}
    #prev_active = []
    prev_orient = None
    prev_ball_owned_team = None
    prev_ball_owned_player = None
    pass_or_not = False
    with open(logfile_path, "a") as f:
        f.writelines("###################################################")
        f.writelines('\n')
        while not done:
            while not done:
                if ball_owned_team == 1:
                    break

                if ball_owned_team != -1:
                    prev_ball_owned_team = ball_owned_team
                    prev_ball_owned_player = ball_owned_player
                #elif prev_obs:
                #    prev_ball_owned_team = prev_obs[0][0]["ball_owned_team"]
                #    prev_ball_owned_player = prev_obs[0][0]["ball_owned_player"]
                    #prev_active = [prev_obs[0]["active"], prev_obs[1]["active"]]
                #if args.opp:
                #    action = agent_opp(obs, args)
                #else:
                #    action, a, attack_att, defence_att, opp_num = agent(obs, args)
                #    active_idx = [obs[0]["active"]]
                #    team_att_idx, opp_att_idx = split_att_def_idx(attack_att, defence_att, active_idx, opp_num)
                #    #team_att_idx, opp_att_idx = split_att_def_idx_(attack_att, defence_att, active_idx)
                state_dict = fe_off.encode(obs)
                state_dict_tensor = stt(state_dict)
                with torch.no_grad():
                    left_att_idx, right_att_idx = model_off(state_dict_tensor)
                    most_att_idx = find_most_att_idx(left_att_idx)

                [obs, opp_obs], [rew, _], done, info = env.att_step([19, 19], [most_att_idx,{"None":None}])
                active = [obs["active"], opp_obs["active"]]
                ball_owned_team = obs["ball_owned_team"]
                ball_owned_player = obs["ball_owned_player"]

                left_active_x, left_active_y = obs["left_team"][active[0]]
                right_active_x, right_active_y = obs["right_team"][active[1]]
                ball_x, ball_y, ball_z = obs["ball"]
                dis_left_to_ball = np.linalg.norm([ball_x-left_active_x, ball_y-left_active_y])
                dis_right_to_ball = np.linalg.norm([ball_x-right_active_x, ball_y-right_active_y])
                dis_to_ball = [dis_left_to_ball, dis_right_to_ball]

                if dis_to_ball[0] < 0.02 and dis_to_ball[0] < dis_to_ball[1]:
                    ball_owned_team = 0
                    ball_owned_player = active[0]
                elif dis_to_ball[1] < 0.02 and dis_to_ball[1] < dis_to_ball[0]:
                    ball_owned_team = 1
                    ball_owned_player = active[1]
            
                if ball_owned_team != -1:
                    prev_obs = obs
                #obs = env.att_step(action, [[],[]])
                left_score = rew

                if left_score != 0:
                    state_dict = {state_list[6]:{prev_ball_owned_team: prev_ball_owned_player}}
                    prev_obs = {}
                    #prev_active = []
                    prev_ball_owned_team = None
                    prev_ball_owned_player = None
                    pass_or_not = False

                elif not prev_obs:
                    state_dict = {state_list[0]: active, "most_att":{0: most_att_idx}} #争球
                elif prev_ball_owned_team == ball_owned_team and prev_ball_owned_player == ball_owned_player:
                    state_dict = {state_list[2]: {ball_owned_team: ball_owned_player, "most_att":{0: most_att_idx}}} #带球
                    pass_or_not = False
                elif prev_ball_owned_team != ball_owned_team and ball_owned_team != -1 and prev_ball_owned_team != None:
                    state_dict = {state_list[4]: [{ball_owned_team: ball_owned_player}, {prev_ball_owned_team: prev_ball_owned_player}], "most_att":{0: most_att_idx}} #断球
                    pass_or_not = False
                elif prev_ball_owned_team == ball_owned_team or (prev_ball_owned_team == None and ball_owned_team != -1):
                    state_dict = {state_list[3]: [{ball_owned_team: ball_owned_player}, {prev_ball_owned_team: prev_ball_owned_player}], "most_att":{0: most_att_idx}} #接传球
                    pass_or_not = False
                elif prev_ball_owned_player == active[prev_ball_owned_team]:
                    state_dict = {state_list[5]: [{prev_ball_owned_team: prev_ball_owned_player}], "most_att":{0: most_att_idx}} #停球
                    pass_or_not = False
                else:
                    if not pass_or_not:
                        state_dict = {state_list[1]: [{prev_ball_owned_team: prev_ball_owned_player}, {prev_ball_owned_team: active[prev_ball_owned_team]}], "most_att":{0: most_att_idx}} #传球
                        pass_or_not = True
                    else:
                        state_dict = {state_list[7]: [{prev_ball_owned_team: prev_ball_owned_player}, {prev_ball_owned_team: active[prev_ball_owned_team]}], "most_att":{0: most_att_idx}} #球飞在空中
            

                left_score += init_left_score
                init_left_score = left_score

                steps += 1
                print("episode:", episode, "step:", steps, "score:", left_score, "state:", state_dict)

                f.writelines(str(["episode:", episode, "step:", steps, "score:", left_score, "state:", state_dict])) 
                f.writelines('\n')
            
            while not done:
                if ball_owned_team == 0:
                    break

                if ball_owned_team != -1:
                    prev_ball_owned_team = ball_owned_team
                    prev_ball_owned_player = ball_owned_player
                #elif prev_obs:
                #    prev_ball_owned_team = prev_obs[0][0]["ball_owned_team"]
                #    prev_ball_owned_player = prev_obs[0][0]["ball_owned_player"]
                    #prev_active = [prev_obs[0]["active"], prev_obs[1]["active"]]
                #if args.opp:
                #    action = agent_opp(obs, args)
                #else:
                #    action, a, attack_att, defence_att, opp_num = agent(obs, args)
                #    active_idx = [obs[0]["active"]]
                #    team_att_idx, opp_att_idx = split_att_def_idx(attack_att, defence_att, active_idx, opp_num)
                #    #team_att_idx, opp_att_idx = split_att_def_idx_(attack_att, defence_att, active_idx)
                state_dict = fe_def.encode(obs)
                state_dict_tensor = stt(state_dict)
                with torch.no_grad():
                    right_att_idx, left_att_idx = model_def(state_dict_tensor)
                    most_att_idx = find_most_att_idx(right_att_idx)

                [obs, opp_obs], [rew, _], done, info = env.att_step([19, 19], [{"None":None}, most_att_idx])
                active = [obs["active"], opp_obs["active"]]
                ball_owned_team = obs["ball_owned_team"]
                ball_owned_player = obs["ball_owned_player"]

                left_active_x, left_active_y = obs["left_team"][active[0]]
                right_active_x, right_active_y = obs["right_team"][active[1]]
                ball_x, ball_y, ball_z = obs["ball"]
                dis_left_to_ball = np.linalg.norm([ball_x-left_active_x, ball_y-left_active_y])
                dis_right_to_ball = np.linalg.norm([ball_x-right_active_x, ball_y-right_active_y])
                dis_to_ball = [dis_left_to_ball, dis_right_to_ball]

                if dis_to_ball[0] < 0.02 and dis_to_ball[0] < dis_to_ball[1]:
                    ball_owned_team = 0
                    ball_owned_player = active[0]
                elif dis_to_ball[1] < 0.02 and dis_to_ball[1] < dis_to_ball[0]:
                    ball_owned_team = 1
                    ball_owned_player = active[1]
            
                if ball_owned_team != -1:
                    prev_obs = obs
                #obs = env.att_step(action, [[],[]])
                left_score = rew

                if left_score != 0:
                    state_dict = {state_list[6]:{prev_ball_owned_team: prev_ball_owned_player}}
                    prev_obs = {}
                    #prev_active = []
                    prev_ball_owned_team = None
                    prev_ball_owned_player = None
                    pass_or_not = False

                elif not prev_obs:
                    state_dict = {state_list[0]: active, "most_att":{1: most_att_idx}} #争球
                elif prev_ball_owned_team == ball_owned_team and prev_ball_owned_player == ball_owned_player:
                    state_dict = {state_list[2]: {ball_owned_team: ball_owned_player}, "most_att":{1: most_att_idx}} #带球
                    pass_or_not = False
                elif prev_ball_owned_team != ball_owned_team and ball_owned_team != -1 and prev_ball_owned_team != None:
                    state_dict = {state_list[4]: [{ball_owned_team: ball_owned_player}, {prev_ball_owned_team: prev_ball_owned_player}], "most_att":{1: most_att_idx}} #断球
                    pass_or_not = False
                elif prev_ball_owned_team == ball_owned_team or (prev_ball_owned_team == None and ball_owned_team != -1):
                    state_dict = {state_list[3]: [{ball_owned_team: ball_owned_player}, {prev_ball_owned_team: prev_ball_owned_player}], "most_att":{1: most_att_idx}} #接传球
                    pass_or_not = False
                elif prev_ball_owned_player == active[prev_ball_owned_team]:
                    state_dict = {state_list[5]: [{prev_ball_owned_team: prev_ball_owned_player}], "most_att":{1: most_att_idx}} #停球
                    pass_or_not = False
                else:
                    if not pass_or_not:
                        state_dict = {state_list[1]: [{prev_ball_owned_team: prev_ball_owned_player}, {prev_ball_owned_team: active[prev_ball_owned_team]}], "most_att":{1: most_att_idx}} #传球
                        pass_or_not = True
                    else:
                        state_dict = {state_list[7]: [{prev_ball_owned_team: prev_ball_owned_player}, {prev_ball_owned_team: active[prev_ball_owned_team]}], "most_att":{1: most_att_idx}} #球飞在空中
            

                left_score += init_left_score
                init_left_score = left_score

                steps += 1
                print("episode:", episode, "step:", steps, "score:", left_score, "state:", state_dict)

                f.writelines(str(["episode:", episode, "step:", steps, "score:", left_score, "state:", state_dict])) 
                f.writelines('\n')
        f.writelines('\n')