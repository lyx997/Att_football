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

def seperate_evaluator(center_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    #fe_module2 = importlib.import_module("encoders." + "encoder_basic")
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    #fe2 = fe_module2.FeatureEncoder()
    #state_to_tensor2 = fe_module2.state_to_tensor
    model_att = center_model[0]
    model_def = center_model[1]
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])

    #
    #env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
    #                                      stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
    #                                      render=False)
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    n_epi = 0
    while True: # episode loop
        seed = random.random()
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0

        ball_owned_team = obs[0]["ball_owned_team"] #-1

        while not done:  # step loop
        
            #attack model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 1: #ball owned by opp change to model_def
                    break

                is_stopped = False
                while signal_queue[0].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    pass
                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_att(state_dict_tensor)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs
                if our_team == 0:
                    obs, rew, done, info = env_left.step([real_action])
                else:
                    obs, rew, done, info = env_right.step([real_action])

                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
                state_prime_dict = fe1.encode(obs[0])
                

                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
    
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
                    summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)

            #defence model
            while not done:  # step loop
                init_t = time.time()

                if ball_owned_team == 0: #ball owned by us so change to model_att
                    break

                is_stopped = False
                while signal_queue[1].qsize() > 0:
                    time.sleep(0.02)
                    is_stopped = True
                if is_stopped:
                    pass
                wait_t += time.time() - init_t
                
                h_in = h_out
                state_dict = fe1.encode(obs[0])
                state_dict_tensor = state_to_tensor1(state_dict, h_in)
                
                t1 = time.time()
                with torch.no_grad():
                    a_prob, m_prob, _, h_out = model_def(state_dict_tensor)
                forward_t += time.time()-t1 
    
                real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
                prev_obs = obs
                if our_team == 0:
                    obs, rew, done, info = env_left.step([real_action])
                else:
                    obs, rew, done, info = env_right.step([real_action])

                ball_owned_team = obs[0]["ball_owned_team"]
                fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
                state_prime_dict = fe1.encode(obs[0])
                
                (h1_in, h2_in) = h_in
                (h1_out, h2_out) = h_out
                state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
    
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
                    summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
                    summary_queue.put(summary_data)

def evaluator(center_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module1 = importlib.import_module("encoders." + arg_dict["encoder"])
    #fe_module2 = importlib.import_module("encoders." + "encoder_basic")
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    #opp_import_model = importlib.import_module("models." + "conv1d_self")
    
    fe1 = fe_module1.FeatureEncoder()
    state_to_tensor1 = fe_module1.state_to_tensor
    #fe2 = fe_module2.FeatureEncoder()
    #state_to_tensor2 = fe_module2.state_to_tensor
    model = center_model
    #opp_model = opp_import_model.Model(arg_dict)
    #opp_model_checkpoint = torch.load(arg_dict["env_evaluation"])
    #opp_model.load_state_dict(opp_model_checkpoint['model_state_dict'])

    #
    #env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
    #                                      stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
    #                                      render=False)
    env_left = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_left"], \
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=0,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    env_right = football_env.create_environment(env_name=arg_dict["env_evaluation"], representation="raw", stacked=False, logdir=arg_dict["log_dir_dump_right"], \
                                          number_of_left_players_agent_controls=0,
                                          number_of_right_players_agent_controls=1,
                                          write_goal_dumps=True, write_full_episode_dumps=False, render=False, write_video=True)
    n_epi = 0
    while True: # episode loop
        seed = random.random()
        if seed < 0.5:
            env_left.reset()   
            obs = env_left.observation()
            our_team = 0
        else:
            env_right.reset()   
            obs = env_right.observation()
            our_team = 1

        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
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
                pass
            wait_t += time.time() - init_t
            
            h_in = h_out
            state_dict = fe1.encode(obs[0])
            state_dict_tensor = state_to_tensor1(state_dict, h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
            forward_t += time.time()-t1 
    
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
    
            prev_obs = obs
            if our_team == 0:
                obs, rew, done, info = env_left.step([real_action])
            else:
                obs, rew, done, info = env_right.step([real_action])

            fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
            state_prime_dict = fe1.encode(obs[0])
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
    
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
                summary_data = (win, score, tot_reward, steps, arg_dict['env_evaluation'], loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

           