import numpy as np
from actor import find_most_att_idx

def calc_reward(rew, prev_obs, obs, player_att_idx, action):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if   (-END_X <= ball_x and ball_x < -PENALTY_X)and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = -2.0
    elif (-PENALTY_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x  and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 1.0
    elif (MIDDLE_X < ball_x   and ball_x <= PENALTY_X) and (-END_Y < ball_y and ball_y < END_Y):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) -  np.sum(prev_obs["left_team_yellow_card"])
    right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
    yellow_r = right_yellow - left_yellow

    
    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0

    active = obs['active']
    change_ball_owned_reward = 0.0
    owned_ball_team_prev = prev_obs["ball_owned_team"]
    owned_ball_team = obs["ball_owned_team"]
    if owned_ball_team == 0 and owned_ball_team_prev == 1 :
        change_ball_owned_reward = 10.0
    elif owned_ball_team == 0 and owned_ball_team_prev == -1 and ball_position_r < 0 and active != 0:
        change_ball_owned_reward = 3.0
    elif owned_ball_team == 1 and owned_ball_team_prev == -1 :
        change_ball_owned_reward = -3.0
    elif owned_ball_team == 1 and owned_ball_team_prev == 0 :
        change_ball_owned_reward = -10.0

    pass_actions = [2, 3, 4]
    attention_reward = 0.0
    if player_att_idx != None:
        most_team_att = find_most_att_idx(player_att_idx, active)
        if active in most_team_att and player_att_idx[0][0][active] > 0.3 and action in pass_actions:
            attention_reward = 1.0

    empty_pass_reward = 0.0
    if player_att_idx != None:
        if  owned_ball_team_prev == 0 and owned_ball_team == -1 and active in most_team_att and active != 0:
            obs_right_team = np.array(obs['right_team'])
            right_team_distance = np.linalg.norm(obs_right_team - obs['left_team'][active], axis=1, keepdims=True)
            right_team_closest_distance = np.min(right_team_distance)

            if right_team_closest_distance > 0.05:
                empty_pass_reward = 1.0

    ball_close_reward = 0
    if owned_ball_team_prev != 0:
        obs_left_team = np.array(obs["left_team"])
        obs_ball = np.array(obs["ball"][:-1])
        left_team_distance = np.linalg.norm(obs_ball - obs_left_team, axis=1, keepdims=True)
        left_team_closest_distance = np.min(left_team_distance)

        if left_team_closest_distance < 0.01:
            ball_close_reward = 1.0

    reward = 5.0*win_reward + 5.0*rew + 0.003*ball_position_r + 0.3*yellow_r  + 0.1*change_ball_owned_reward + 0.1*empty_pass_reward + 0.01*ball_close_reward
        
    return reward