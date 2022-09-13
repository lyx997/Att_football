from cv2 import _OutputArray_DEPTH_MASK_FLT
import gfootball.env as football_env
import time, pprint, importlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta


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

def sup_evaluator(center_model, signal_queue, summary_queue, arg_dict):

    print("Evaluator process started")
    fe_rl_module = importlib.import_module("sup_encoders." + arg_dict["encoder_rl"])
    fe_rw_module = importlib.import_module("sup_encoders." + arg_dict["encoder_rw"])
    rewarder = importlib.import_module("sup_rewarders." + arg_dict["rewarder"])
    imported_rw_model = importlib.import_module("sup_models." + arg_dict["rw_off_model"])
    imported_rl_model = importlib.import_module("sup_models." + arg_dict["rl_model"])
    
    fe_rl = fe_rl_module.FeatureEncoder()
    fe_rw = fe_rw_module.FeatureEncoder()
    rl_state_to_tensor = fe_rl_module.state_to_tensor
    rw_state_to_tensor = fe_rw_module.state_to_tensor
    
    cpu_device = torch.device('cpu')
    rw_checkpoint = torch.load(arg_dict["rew_model_off_path"], map_location=cpu_device)
    rew_model = imported_rw_model.Model(arg_dict)
    rew_model.load_state_dict(rw_checkpoint['model_state_dict'])

    rl_model = imported_rl_model.Model(arg_dict)
    rl_model.load_state_dict(center_model.state_dict())

    env_left = football_env.create_environment(env_name=arg_dict['env_evaluation'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    env_right = football_env.create_environment(env_name=arg_dict['env_evaluation'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    n_epi = 0
    rollout = []
    while True: # episode loop
        #seed = random.random()
        seed = 0.1
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        prev_obs = [[]]
        prev_ball_owned_team = None 
        att_rew = 0

        done = False
        steps, score, tot_reward, tot_good_pass, win = 0, 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        
        while not done:  # step loop
            init_t = time.time()
            
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                rl_model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            rl_state_dict = fe_rl.encode(obs[0])
            rl_state_dict_tensor = rl_state_to_tensor(rl_state_dict, h_in)

            rw_state_dict, opp_num = fe_rw.encode(obs[0])
            rw_state_dict_tensor = rw_state_to_tensor(rw_state_dict)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = rl_model(rl_state_dict_tensor)

                if prev_ball_owned_team == 0:
                    left_att_idx, right_att_idx = rew_model(rw_state_dict_tensor)

            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)


            if our_team == 0:
                obs, rew, done, info = env_left.att_step(real_action,[[],[],[]])
            else:
                obs, rew, done, info = env_right.att_step(real_action,[[],[],[]])

            active = obs[0]["active"]

            if prev_ball_owned_team == 0:
                att_rew = float(left_att_idx[0,active])
            else:
                att_rew = 0

            rew=rew[0]
            if rew != 0:
                prev_obs = [[]]

            fin_r, good_pass_counts = rewarder.calc_reward(rew, att_rew, prev_obs[0], obs[0])
            state_prime_dict = fe_rl.encode(obs[0])

            if obs[0]["ball_owned_team"] != -1:
                prev_obs = obs
                prev_ball_owned_team = obs[0]["ball_owned_team"]
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            rl_state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

            steps += 1
            score += rew
            tot_reward += fin_r
            tot_good_pass += good_pass_counts
            
            loop_t += time.time()-init_t
            
            if done:
                if score > 0:
                    win = 1
                if our_team == 0:
                    print("model in left evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                else:
                    print("model in right evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, tot_good_pass, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def evaluator(center_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    #fe_module2 = importlib.import_module("encoders." + "encoder_basic")
    rewarder = importlib.import_module("sup_rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    #fe2 = fe_module2.FeatureEncoder()
    #state_to_tensor2 = fe_module2.state_to_tensor
    model = center_model
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False, write_video=False)
    
    n_epi = 0
    while True: # episode loop
        #seed = random.random()
        seed = 0.1
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1
        
        prev_obs = [[]]
        done = False
        steps, score, tot_reward, tot_good_pass, win = 0, 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        #opp_h_out = (torch.zeros([1, 1, 256], dtype=torch.float), 
        #        torch.zeros([1, 1, 256], dtype=torch.float))

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        
        while not done:  # step loop
            init_t = time.time()
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                #model.load_state_dict(center_model.state_dict())
                pass
            wait_t += time.time() - init_t
            
            h_in = h_out
            #opp_h_in = opp_h_out
            state_dict = fe1.encode(obs[0])
            #opp_state_dict = fe2.encode(obs[1])
            state_dict_tensor = state_to_tensor1(state_dict, h_in)
            #opp_state_dict_tensor = state_to_tensor2(opp_state_dict, opp_h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
                #opp_a_prob, opp_m_prob, _, opp_h_out = opp_model(opp_state_dict_tensor)
            forward_t += time.time()-t1 

            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
            #opp_real_action, opp_a, opp_m, opp_need_m, opp_prob, opp_prob_selected_a, opp_prob_selected_m = get_action(opp_a_prob, opp_m_prob)

            if our_team == 0:
                obs, rew, done, info = env_left.att_step([real_action],[[],[],[]])
            else:
                obs, rew, done, info = env_right.att_step([real_action],[[],[],[]])

            rew = rew[0]
            if rew != 0:
                prev_obs = [[]]

            fin_r, good_pass_counts = rewarder.calc_reward(rew, 0, prev_obs[0], obs[0])
            state_prime_dict = fe1.encode(obs[0])
            
            if obs[0]["ball_owned_team"] != -1:
                prev_obs = obs

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

            steps += 1
            score += rew
            tot_reward += fin_r
            tot_good_pass += good_pass_counts
            
            loop_t += time.time()-init_t
            
            if done:
                if score > 0:
                    win = 1

                if our_team == 0:
                    print("model in left evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                else:
                    print("model in right evaluate with ", arg_dict["env_evaluation"]," model: score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, tot_good_pass, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

