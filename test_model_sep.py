import gfootball.env as football_env

import time, pprint, json, os, importlib, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from evaluator_with_hard_att_def import seperate_evaluator

def find_most_att_idx(player_att2_idx, active_idx):
    most_att_idx = player_att2_idx.sort(descending=True)[1][0]
    if most_att_idx >= active_idx:
        most_att_idx += 1
    return most_att_idx

def get_action(a_prob, m_prob):
    a = Categorical(a_prob).sample().item()
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0
    if a==0:
        real_action = a
        prob = prob_selected_a
    elif a==1:
        m = Categorical(m_prob).sample().item()
        need_m = 1
        real_action = m + 1
        prob_selected_m = m_prob[0][0][m].item()
        prob = prob_selected_a* prob_selected_m
    else:
        real_action = a + 7
        prob = prob_selected_a

    assert prob != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a,m,prob_selected_a,prob_selected_m)
    
    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m

def seperate_evaluator(center_model, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder_att"])
    fe_module2 = importlib.import_module("encoders." + arg_dict["encoder_def"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    fe2 = fe_module2.FeatureEncoder()
    state_to_tensor2 = fe_module2.state_to_tensor
    model_att = center_model[0]
    model_def = center_model[1]
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])

    #
    #env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
    #                                      stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
    #                                      render=False)
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=True, render=arg_dict["render"], write_video=arg_dict["write_video"])
    env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=True, render=False, write_video=True)
    n_epi = 0
    while True: # episode loop
        seed = 0.1
        #seed = random.random()
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        done = False
        episode, steps, score, tot_reward, win = 1, 1, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0

        ball_owned_team = obs[0]["ball_owned_team"] #-1
        ball_x = obs[0]["ball"][0]

        while not done:  # step loop
        
            #attack model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 1 and ball_x < 0.: #ball owned by opp change to model_def
                    break

                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out, player_att_idx = model_att(state_dict_tensor)
                    active_idx = obs[0]["active"]
                    most_att_idx = find_most_att_idx(player_att_idx[2], active_idx)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs
                if our_team == 0:
                    obs, rew, done, info = env_left.att_step([real_action], [])
                else:
                    obs, rew, done, info = env_right.att_step([real_action], [])

                rew = rew[0]
                ball_owned_team = obs[0]["ball_owned_team"]
                ball_x = obs[0]["ball"][0]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], False)
                state_prime_dict = fe1.encode(obs[0])
                
                print("episode", episode, "step:", steps, "score:", rew)

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
    
                steps += 1
                score += rew
                tot_reward += fin_r
                
                loop_t += time.time()-init_t
                
                if done:
                    if score > 0:
                        win = 1

                    if our_team == 0:
                        print("model in left evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    else:
                        print("model in right evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)

            #defence model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 0 or ball_x >= 0.: #ball owned by us so change to model_att
                    break

                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe2.encode(obs[0])
                state_dict_tensor = state_to_tensor2(state_dict, h_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out, [] = model_def(state_dict_tensor)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs
                if our_team == 0:
                    obs, rew, done, info = env_left.att_step([real_action], [])
                else:
                    obs, rew, done, info = env_right.att_step([real_action], [])
                

                rew = rew[0]
                ball_owned_team = obs[0]["ball_owned_team"]
                ball_x = obs[0]["ball"][0]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], False)
                state_prime_dict = fe2.encode(obs[0])

                print("episode", episode, "step:", steps, "score:", rew)
                
                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
    
                steps += 1
                score += rew
                tot_reward += fin_r
                
                loop_t += time.time()-init_t
                
                if done:
                    if score > 0:
                        win = 1
                    if our_team == 0:
                        print("model in left evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                    else:
                        print("model in right evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)

        episode += 1

arg_dict = {
        "render": False,
        "write_video": True,
        "log_dir": 'test_dump',
        "lstm_size": 256,
        "learning_rate" : 0.0001,
        "attack_trained_model_path" : 'logs/[06-28]09.53.57_gat_conv_seperate_rewarder_att_def8/att/model_att_44414208.tar', # use when you want to continue traning from given model.
        "defence_trained_model_path" : 'logs/[06-28]09.53.57_gat_conv_seperate_rewarder_att_def8/def/model_def_20406528.tar', # use when you want to continue traning from given model.

        "encoder_att" : "encoder_gat_att_def_latest4",
        "encoder_def" : "encoder_basic",
        "rewarder" : "rewarder_att_def15",
        "model_att" : "gat_att_def6_latest5_att",
        "model_def" : "conv1d_def",
        "algorithm" : "ppo_with_lstm",

        "env_evaluation":'11_vs_11_competition'  # for evaluation of self-play trained agent (like validation set in Supervised Learning)
    }

if not os.path.exists(arg_dict["log_dir"]):
    os.mkdir(arg_dict["log_dir"])

fe_att = importlib.import_module("encoders." + arg_dict["encoder_att"])
fe_att = fe_att.FeatureEncoder()
arg_dict["att_feature_dims"] = fe_att.get_feature_dims()
fe_def = importlib.import_module("encoders." + arg_dict["encoder_def"])
fe_def = fe_def.FeatureEncoder()
arg_dict["def_feature_dims"] = fe_def.get_feature_dims()
model_att = importlib.import_module("models." + arg_dict["model_att"])
model_def = importlib.import_module("models." + arg_dict["model_def"])
cpu_device = torch.device('cpu')
center_model_att = model_att.Model(arg_dict)
center_model_def = model_def.Model(arg_dict)
attack_model_checkpoint = torch.load(arg_dict["attack_trained_model_path"], map_location=cpu_device)
center_model_att.load_state_dict(attack_model_checkpoint['model_state_dict'])
defence_model_checkpoint = torch.load(arg_dict["defence_trained_model_path"], map_location=cpu_device)
center_model_def.load_state_dict(defence_model_checkpoint['model_state_dict'])

center_model = [center_model_att, center_model_def]
seperate_evaluator(center_model, arg_dict)