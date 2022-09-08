import numpy as np
import math

def calc_reward(rew, prev_obs, obs, prev_most_att_idx, prev_most_att, highpass, prev_opp_most_att_idx, prev_opp_most_att):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if   (-END_X <= ball_x and ball_x < -PENALTY_X)and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = -2.0
    elif (-PENALTY_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x  and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = 1.0
    elif (MIDDLE_X < ball_x   and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0
    
    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0
        elif my_score < opponent_score:
            win_reward = -1.0

    yellow_r = 0.0
    change_ball_owned_reward = 0.0
    safe_pass_reward = 0.0
    attention_reward = 0.0
    good_pass_counts = 0
    att_pass_counts = 0
    active = obs['active']
    active_tired = obs['left_team_tired_factor'][active]
    tired_reward = 0
    
    #if prev_most_att_idx:
    #    if active in prev_most_att_idx and highpass:
    #       #att_high_pass_counts = 1
    #       #attention_reward = float(np.exp(prev_most_att))
    #       attention_reward = 1

    if prev_obs:

        yellow_r = np.sum(prev_obs["left_team_yellow_card"]) - np.sum(obs["left_team_yellow_card"]) 
        #right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
        #yellow_r = right_yellow - left_yellow

        prev_active = prev_obs['active']
        prev_active_tired = prev_obs['left_team_tired_factor'][prev_active]

        #if active == prev_active and (PENALTY_X > ball_x or -PENALTY_Y > ball_y or ball_y > PENALTY_Y):
        #    tired_reward = prev_active_tired - active_tired

        prev_ball_x = prev_obs["ball"][0]
        prev_owned_ball_team = prev_obs["ball_owned_team"]
        prev_owned_ball_player = prev_obs["ball_owned_player"]
        prev_obs_right_team_x = np.array(prev_obs['right_team'][1:][0])
        prev_max_right_team_x = float(np.min(prev_obs_right_team_x))
        owned_ball_team = obs["ball_owned_team"]
        owned_ball_player = obs["ball_owned_player"]
        active_pos_x = obs['left_team'][active][0]
        if owned_ball_team == 0 and prev_owned_ball_team == 1 :
            change_ball_owned_reward = 3.0
        elif owned_ball_team == 1 and prev_owned_ball_team == 0 :
            change_ball_owned_reward = -3.0
        elif owned_ball_team == 1 and prev_owned_ball_team == 1 and prev_owned_ball_player != owned_ball_player and owned_ball_player in prev_opp_most_att_idx:
            change_ball_owned_reward = float(np.exp(prev_opp_most_att)) - 2
        elif owned_ball_team == 1 and prev_owned_ball_team == 1 and prev_owned_ball_player != owned_ball_player:
            change_ball_owned_reward = -1.0
        elif owned_ball_team == 0 and prev_owned_ball_team == 0 and ball_x - prev_ball_x > -0.1 and prev_ball_x < prev_max_right_team_x and \
            owned_ball_player != 0 and prev_owned_ball_player != owned_ball_player and owned_ball_player in prev_most_att_idx:
            good_pass_counts = 1
            #change_ball_owned_reward = float(np.exp(prev_most_att - 0.3))
        elif owned_ball_team == 0 and prev_owned_ball_team == 0 and owned_ball_player == prev_owned_ball_player and (PENALTY_X > ball_x or -PENALTY_Y > ball_y or ball_y > PENALTY_Y):
            tired_reward = prev_active_tired - active_tired

        #prev_active_pos_x = obs['left_team'][prev_active][0]
        if  prev_owned_ball_team == 0 and owned_ball_team == 0 and owned_ball_player != 0 and prev_owned_ball_player != owned_ball_player:
            obs_right_team = np.array(obs['right_team'])
            prev_obs_right_team = np.array(prev_obs['right_team'])
            right_team_distance = np.linalg.norm(obs_right_team - obs['left_team'][active], axis=1, keepdims=True)
            right_team_closest_distance = np.min(right_team_distance)
            prev_right_team_distance = np.linalg.norm(prev_obs_right_team - prev_obs['left_team'][prev_active], axis=1, keepdims=True)
            prev_right_team_closest_distance = np.min(prev_right_team_distance)

            if right_team_closest_distance - prev_right_team_closest_distance > 0.01 and ball_x - prev_ball_x > -0.1:
                #safe_pass_reward = 2.0
                good_pass_counts = 1
            #else:
            #    safe_pass_reward = 1.0
            #    good_pass_counts = 1
            #elif right_team_closest_distance - prev_right_team_closest_distance > 0.0:
            #    safe_pass_reward = 0.0
            #else:
            #    safe_pass_reward = -1.0

    reward = 5.0*win_reward + 5.0*rew + 0.1*yellow_r + 0.001*ball_position_r + 0.1*change_ball_owned_reward + 10*tired_reward 
        
    return reward, good_pass_counts