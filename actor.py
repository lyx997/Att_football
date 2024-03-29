import gfootball.env as football_env
import time, pprint, importlib, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import math

from datetime import datetime, timedelta

def find_most_att_idx(player_att2_idx):
    player_att2_idx = player_att2_idx.squeeze()
    player_most_att_idx = player_att2_idx.sort(descending=True)[1][0]
    return int(player_most_att_idx)

def split_att_idx(all_sorted_idx):
    team_att_idx_list = []
    opp_att_idx_list = []
    for idx in all_sorted_idx:
        if idx > 10:
            opp_att_idx_list.append(idx % 11)
        else:
            team_att_idx_list.append(idx)
    
    return team_att_idx_list, opp_att_idx_list

def epsilon_get_action(q_a, action_prob, epsilon):

    if np.random.uniform() > epsilon:
        a = torch.max(q_a, 2)[1].item()
    else:
        a = Categorical(action_prob).sample().item()

    return a


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

def on_policy_actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())

    
    env_left = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    env_right = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout = []
    
    while True: # episode loop
        seed = arg_dict["seed"]
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        prev_obs = [[]]
        prev_most_att_idx = []
        prev_most_att = 0.0
        prev_opp_most_att_idx = []
        prev_opp_most_att = 0.0
        highpass=False
        done = False
        get_score = False
        steps, score, tot_reward, tot_good_pass, win= 0, 0, 0, 0, 0
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
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            state_dict, opp_num = fe.encode(obs[0], seed)
            state_dict_tensor = state_to_tensor(state_dict, h_in)

            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out, player_att_idx, opp_att_idx = model(state_dict_tensor)
                active_idx = obs[0]["active"]
                prev_most_att_idx, prev_opp_most_att_idx, prev_most_att, prev_opp_most_att = find_most_att_idx(player_att_idx[0], opp_att_idx[0], active_idx, opp_num)

            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)

            if a == 3:
                highpass=True

            if our_team == 0:
                obs, rew, done, info = env_left.att_step(real_action, [[], []])
            else:
                obs, rew, done, info = env_right.att_step(real_action, [[], []])
            rew = rew[0]
            if rew != 0:
                #get_score = True
                prev_obs = [[]]
                prev_most_att_idx = []
                prev_opp_most_att_idx = []

            fin_r, good_pass_counts = rewarder.calc_reward(rew, prev_obs[0], obs[0], prev_most_att_idx, prev_most_att, highpass, prev_opp_most_att_idx, prev_opp_most_att)
            state_prime_dict, opp_num = fe.encode(obs[0], seed)

            if obs[0]["ball_owned_team"] != -1:
                if not prev_obs[0]:
                    highpass = False
                elif prev_obs[0]["active"] != obs[0]["active"]:
                    highpass = False
                prev_obs = obs

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            tot_good_pass += good_pass_counts

            loop_t += time.time()-init_t

            if done:
                if score > 0:
                    win = 1
                if our_team == 0:
                    print("model in left score",score,"total reward",tot_reward)
                else:
                    print("model in right score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, tot_good_pass, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def off_policy_actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())

    
    env_left = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    env_right = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout = []
    EPS_START = arg_dict["epsilon_start"]
    EPS_END = arg_dict["epsilon_end"]
    EPS_DECAY = arg_dict['epsilon_decay']
    episode_count = 0

    actions = torch.ones((1,19), dtype=torch.float32)
    action_prob = F.softmax(actions, dim=-1)

    while True: # episode loop
        seed = 0.1
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        done = False
        steps, score, tot_reward, win= 0, 0, 0, 0
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
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = state_to_tensor(state_dict, h_in)

            t1 = time.time()
            with torch.no_grad():
                #a_prob, m_prob, _, h_out, player_att_idx = model(state_dict_tensor)
                q_a, h_out, player_att_idx = model(state_dict_tensor)
                active_idx = obs[0]["active"]
                most_att_idx = find_most_att_idx(player_att_idx[2], active_idx)

            forward_t += time.time()-t1 
            #real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode_count / EPS_DECAY)
            real_action = epsilon_get_action(q_a, action_prob, epsilon)

            prev_obs = obs

            if our_team == 0:
                obs, rew, done, info = env_left.att_step(real_action, [[], []])
            else:
                obs, rew, done, info = env_right.att_step(real_action, [[], []])

            rew = rew[0]
            fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], most_att_idx)
            state_prime_dict = fe.encode(obs[0])

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            #transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            transition = (state_dict, real_action, fin_r, state_prime_dict, done)

            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r

            loop_t += time.time()-init_t

            if done:
                episode_count += 1
                if score > 0:
                    win = 1
                if our_team == 0:
                    print("model in left score",score,"total reward",tot_reward, 'epsilon', epsilon)
                else:
                    print("model in right score",score,"total reward",tot_reward, 'epsilon', epsilon)
                summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def seperate_actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module_att = importlib.import_module("encoders." + arg_dict["encoder_att"])
    fe_module_def = importlib.import_module("encoders." + arg_dict["encoder_def"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model_att = importlib.import_module("models." + arg_dict["model_att"])
    imported_model_def = importlib.import_module("models." + arg_dict["model_def"])
    
    fe_att = fe_module_att.FeatureEncoder()
    att_state_to_tensor = fe_module_att.state_to_tensor
    fe_def = fe_module_def.FeatureEncoder()
    def_state_to_tensor = fe_module_def.state_to_tensor
    
    model_att = imported_model_att.Model(arg_dict)
    model_att.load_state_dict(center_model[0].state_dict())

    model_def = imported_model_def.Model(arg_dict)
    model_def.load_state_dict(center_model[1].state_dict())
    
    env_left = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    env_right = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout_att = []
    rollout_def = []
    
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
        steps, score, tot_reward, win= 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        ball_owned_team = obs[0]["ball_owned_team"] #-1
        ball_x = obs[0]["ball"][0]

        while not done:

            #attack model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 1 and ball_x < 0.: #ball owned by opp change to model_def
                    break

                is_stopped = False
                while signal_queue[0].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    model_att.load_state_dict(center_model[0].state_dict())
                wait_t += time.time() - init_t

                h_in = h_out
                state_dict = fe_att.encode(obs[0])
                state_dict_tensor = att_state_to_tensor(state_dict, h_in)

                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out, player_att_idx, _ = model_att(state_dict_tensor)
                    active_idx = obs[0]["active"]
                    most_att_idx = find_most_att_idx(player_att_idx[2], active_idx)
                forward_t += time.time()-t1 
                real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)

                prev_obs = obs

                if our_team == 0:
                    obs, rew, done, info = env_left.att_step(real_action, [])
                else:
                    obs, rew, done, info = env_right.att_step(real_action, [])

                rew = rew[0]
                ball_owned_team = obs[0]["ball_owned_team"]
                ball_x = obs[0]["ball"][0]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], most_att_idx)
                state_prime_dict = fe_att.encode(obs[0])

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

                if ball_owned_team == 1 and ball_x < 0.:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, True, need_m) #change to model defence
                else:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

                rollout_att.append(transition)
                if len(rollout_att) == arg_dict["rollout_len"]:
                    data_queue[0].put(rollout_att)
                    rollout_att = []
                    model_att.load_state_dict(center_model[0].state_dict())

                steps += 1
                score += rew
                tot_reward += fin_r

                loop_t += time.time()-init_t

                if done:
                    if score > 0:
                        win = 1
                    if our_team == 0:
                        print("model in left score",score,"total reward",tot_reward)
                    else:
                        print("model in right score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)


             
            #defence model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 0 or ball_x >= 0.: #ball owned by us so change to model_att
                    break

                is_stopped = False
                while signal_queue[1].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    model_def.load_state_dict(center_model[1].state_dict())
                wait_t += time.time() - init_t

                h_in = h_out
                state_dict = fe_def.encode(obs[0])
                state_dict_tensor = def_state_to_tensor(state_dict, h_in)

                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out, _ = model_def(state_dict_tensor)
                    #most_att_idx = find_most_att_idx(player_def_idx[2], 20)
                forward_t += time.time()-t1 
                real_action, a, m, need_m, prob, _, _ = get_action(a_prob, m_prob)

                prev_obs = obs

                if our_team == 0:
                    obs, rew, done, info = env_left.att_step(real_action, [])
                else:
                    obs, rew, done, info = env_right.att_step(real_action, [])

                rew = rew[0]
                ball_owned_team = obs[0]["ball_owned_team"]
                ball_x = obs[0]["ball"][0]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0], False)
                state_prime_dict = fe_def.encode(obs[0])

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

                if ball_owned_team == 0 or ball_x >= 0.:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, True, need_m) # change to model attack
                else:
                    transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)

                rollout_def.append(transition)
                if len(rollout_def) == arg_dict["rollout_len"]:
                    data_queue[1].put(rollout_def)
                    rollout_def = []
                    model_def.load_state_dict(center_model[1].state_dict())

                steps += 1
                score += rew
                tot_reward += fin_r

                loop_t += time.time()-init_t

                if done:
                    if score > 0:
                        win = 1
                    if our_team == 0:
                        print("model in left score",score,"total reward",tot_reward)
                    else:
                        print("model in right score",score,"total reward",tot_reward)
                    summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)

def sup_rl_actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_rl_module = importlib.import_module("sup_encoders." + arg_dict["encoder_rl"])
    fe_rw_module = importlib.import_module("sup_encoders." + arg_dict["encoder_rw"])
    rewarder = importlib.import_module("sup_rewarders." + arg_dict["rewarder"])
    imported_rw_off_model = importlib.import_module("sup_models." + arg_dict["rw_off_model"])
    imported_rw_def_model = importlib.import_module("sup_models." + arg_dict["rw_def_model"])
    imported_rl_model = importlib.import_module("sup_models." + arg_dict["rl_model"])
    
    fe_rl = fe_rl_module.FeatureEncoder()
    fe_rw = fe_rw_module.FeatureEncoder()
    rl_state_to_tensor = fe_rl_module.state_to_tensor
    rw_state_to_tensor = fe_rw_module.state_to_tensor
    
    cpu_device = torch.device('cpu')
    rw_off_checkpoint = torch.load(arg_dict["rew_model_off_path"], map_location=cpu_device)
    rew_off_model = imported_rw_off_model.Model(arg_dict)
    rew_off_model.load_state_dict(rw_off_checkpoint['model_state_dict'])

    rw_def_checkpoint = torch.load(arg_dict["rew_model_def_path"], map_location=cpu_device)
    rew_def_model = imported_rw_def_model.Model(arg_dict)
    rew_def_model.load_state_dict(rw_def_checkpoint['model_state_dict'])

    rl_model = imported_rl_model.Model(arg_dict)
    rl_model.load_state_dict(center_model.state_dict())

    env_left = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    env_right = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
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
        ball_x = obs[0]["ball"][0]
        opp_num = None

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
            rw_state_dict = fe_rw.encode(obs[0])
            rl_state_dict_tensor = rl_state_to_tensor(rl_state_dict, h_in)
            rw_state_dict_tensor = rw_state_to_tensor(rw_state_dict)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = rl_model(rl_state_dict_tensor)

                if prev_ball_owned_team == 0 and ball_x > -0.2:
                    left_att_idx, _ = rew_off_model(rw_state_dict_tensor)
                elif prev_ball_owned_team == 1:
                    right_att_idx, _ = rew_def_model(rw_state_dict_tensor)
                    opp_num = find_most_att_idx(right_att_idx)

            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)


            if our_team == 0:
                obs, rew, done, info = env_left.step(real_action)
            else:
                obs, rew, done, info = env_right.step(real_action)

            active = obs[0]["active"]

            if prev_ball_owned_team == 0 and ball_x > -0.2:
                att_rew = float(left_att_idx[0,active])
                if att_rew < 0.3:
                    att_rew = 0
            #elif prev_ball_owned_team == 1:
            #    att_rew = -0.1*float(right_att_idx[0,opp_num])
            #    #if att_rew > -0.15:
            #    #    att_rew = 0
            else:
                att_rew = 0

            #rew=rew[0]
            if rew != 0:
                prev_obs = [[]]

            fin_r, good_pass_counts = rewarder.calc_reward(rew, att_rew, prev_obs[0], obs[0], opp_num)
            state_prime_dict = fe_rl.encode(obs[0])

            ball_x = obs[0]["ball"][0]
            if obs[0]["ball_owned_team"] != -1:
                prev_obs = obs
                prev_ball_owned_team = obs[0]["ball_owned_team"]
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            rl_state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (rl_state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                rl_model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            tot_good_pass += good_pass_counts
            
            loop_t += time.time()-init_t
            
            if done:
                if score > 0:
                    win = 1
                if our_team == 0:
                    print("model in left score",score,"total reward",tot_reward)
                else:
                    print("model in right score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, tot_good_pass, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)
            
def actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("sup_rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())

    env_left = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    env_right = football_env.create_environment(env_name=arg_dict['env'], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
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
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t
            
            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = state_to_tensor(state_dict, h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

            if our_team == 0:
                obs, rew, done, info = env_left.step(real_action)
            else:
                obs, rew, done, info = env_right.step(real_action)

            #rew=rew[0]
            if rew != 0:
                prev_obs = [[]]

            fin_r, good_pass_counts = rewarder.calc_reward(rew, 0, prev_obs[0], obs[0])
            state_prime_dict = fe.encode(obs[0])


            if obs[0]["ball_owned_team"] != -1:
                prev_obs = obs
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            tot_good_pass += good_pass_counts
            
            loop_t += time.time()-init_t
            
            if done:
                if score > 0:
                    win = 1
                if our_team == 0:
                    print("model in left score",score,"total reward",tot_reward)
                else:
                    print("model in right score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, tot_good_pass, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def select_opponent(arg_dict):
    onlyfiles_lst = [f for f in listdir(arg_dict["log_dir"]) if isfile(join(arg_dict["log_dir"], f))]
    model_num_lst = []
    for file_name in onlyfiles_lst:
        if file_name[:6] == "model_":
            model_num = file_name[6:]
            model_num = model_num[:-4]
            model_num_lst.append(int(model_num))
    model_num_lst.sort()
            
    coin = random.random()
    if coin<arg_dict["latest_ratio"]:
        if len(model_num_lst) > arg_dict["latest_n_model"]:
            opp_model_num = random.randint(len(model_num_lst)-arg_dict["latest_n_model"],len(model_num_lst)-1)
        else:
            opp_model_num = len(model_num_lst)-1
    else:
        opp_model_num = random.randint(0,len(model_num_lst)-1)
        
    model_name = "/model_"+str(model_num_lst[opp_model_num])+".tar"
    opp_model_path = arg_dict["log_dir"] + model_name
    return opp_model_num, opp_model_path
                
                
def actor_self(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    print("Actor process {} started".format(actor_num))
    cpu_device = torch.device('cpu')
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    state_to_tensor = fe_module.state_to_tensor
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())
    opp_model = imported_model.Model(arg_dict)
    
    env_left = football_env.create_environment(env_name="11_vs_11_stochastic", number_of_right_players_agent_controls=1, representation="raw", \
                                          stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
                                          render=False)
    env_right = football_env.create_environment(env_name="11_vs_11_stochastic", number_of_right_players_agent_controls=1, representation="raw", \
                                          stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
                                          render=False)
    n_epi = 0
    rollout = []
    while True: # episode loop
        #seed = random.random()
        seed = 0.1
        opp_model_num, opp_model_path = select_opponent(arg_dict)
        checkpoint = torch.load(opp_model_path, map_location=cpu_device)
        opp_model.load_state_dict(checkpoint['model_state_dict'])
        print("Current Opponent model Num:{}, Path:{} successfully loaded".format(opp_model_num, opp_model_path))
        del checkpoint

        if seed < 0.5: 
            env_left.reset()   
            [obs, opp_obs] = env_left.observation()
        else:
            env_right.reset()
            [opp_obs, obs] = env_right.observation()

        prev_obs = []
        prev_most_att_idx = []
        prev_opp_most_att_idx = []
        prev_most_att = 0.0
        prev_opp_most_att = 0.0
        highpass = False
        done = False
        active_idx = obs["active"]
        opp_active = opp_obs["active"]
        steps, score, tot_reward, win, tot_good_pass = 0, 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        opp_h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        
        while not done:  # step loop
            init_t = time.time()
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t
            
            h_in = h_out
            opp_h_in = opp_h_out
            state_dict, _ = fe.encode(obs)
            state_dict_tensor = state_to_tensor(state_dict, h_in)
            opp_state_dict, _ = fe.encode(opp_obs, 1-seed)
            opp_state_dict_tensor = state_to_tensor(opp_state_dict, opp_h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out, player_att_idx, opp_att_idx = model(state_dict_tensor)
                #active_idx = obs["active"]
                #prev_most_att_idx, prev_opp_most_att_idx, prev_most_att, prev_opp_most_att = find_most_att_idx(player_att_idx[0], opp_att_idx[0], active_idx, opp_num)
                #most_att_idx = find_most_att_idx(player_att_idx[2], active_idx)
                opp_a_prob, opp_m_prob, _, opp_h_out, _, _ = opp_model(opp_state_dict_tensor)
            forward_t += time.time()-t1 
            
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
            opp_real_action, _, _, _, _, _, _ = get_action(opp_a_prob, opp_m_prob)

            
            if a == 3:
                highpass = True

            if seed < 0.5:
                [obs, opp_obs], [rew, _], done, info = env_left.step([real_action, opp_real_action])
            else:
                [opp_obs, obs], [_, rew], done, info = env_right.step([opp_real_action, real_action])

            #opp_active = opp_obs["active"]

            if rew != 0:
                get_score = True
                prev_obs = []
                prev_most_att_idx = []
                prev_opp_most_att_idx = []

            fin_r, good_pass_counts = rewarder.calc_reward(rew, prev_obs, obs, prev_most_att_idx, prev_most_att, highpass, prev_opp_most_att_idx, prev_opp_most_att)
            state_prime_dict, opp_num = fe.encode(obs, seed)

            if obs["ball_owned_team"] != -1:
                if not prev_obs:
                    highpass = False
                elif prev_obs["active"] != obs["active"]:
                    highpass = False
                prev_obs = obs
                active_idx = obs["active"]
                prev_most_att_idx, prev_opp_most_att_idx, prev_most_att, prev_opp_most_att = find_most_att_idx(player_att_idx[0], opp_att_idx[0], active_idx, opp_num)
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)

            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            tot_good_pass += good_pass_counts
            
            loop_t += time.time()-init_t

            if done:
                if score > 0:
                    win = 1
                if seed < 0.5:
                    print("left score {}, total reward {:.2f}, opp num:{}, right opp:{} ".format(score,tot_reward,opp_model_num, opp_model_path))
                else:
                    print("right score {}, total reward {:.2f}, opp num:{}, left opp:{} ".format(score,tot_reward,opp_model_num, opp_model_path))
                summary_data = (win, score, tot_reward, tot_good_pass, steps, str(opp_model_num), loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)                

