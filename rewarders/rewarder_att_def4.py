import numpy as np

def calc_reward(rew, prev_obs, obs, most_team_att, attack_or_not):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if   (-END_X <= ball_x    and ball_x < -PENALTY_X)and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = -2.0
    elif (-END_X <= ball_x    and ball_x < -MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x  and ball_x <=END_X)     and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x   and ball_x <=END_X)     and (-END_Y < ball_y     and ball_y < END_Y):
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

    change_ball_owned_reward = 0.0
    owned_ball_team_prev = prev_obs["ball_owned_team"]
    owned_ball_team = obs["ball_owned_team"]
    if owned_ball_team == 0 and owned_ball_team_prev == -1 :
        change_ball_owned_reward = 1.0
    elif owned_ball_team == 0 and owned_ball_team_prev == 1 :
        change_ball_owned_reward = 10.0

    elif owned_ball_team == 1 and owned_ball_team_prev == -1 :
        change_ball_owned_reward = -1.0
    elif owned_ball_team == 1 and owned_ball_team_prev == 0 :
        change_ball_owned_reward = -10.0

    ball_not_owned_reward = 0.0
    if owned_ball_team == 1:
        ball_not_owned_reward = -1.0

    attention_reward = 0.0
    #attack model reward
    if attack_or_not:
        if obs['active'] == int(most_team_att):
            attention_reward = 1.0
        
        #reward = 5.0*win_reward + 5.0*rew + 0.003*ball_position_r + 0.3*yellow_r + 0.03*change_ball_owned_reward + 0.1*attention_reward
        reward = 1.0*win_reward + 5.0*rew + 0.003*ball_position_r + 0.3*yellow_r + 0.1*change_ball_owned_reward + 0.1*attention_reward

    #defence model reward
    else:
        player_num = obs['active']
        ball_x, ball_y, ball_z = obs['ball']
        player_pos_x, player_pos_y = obs['left_team'][player_num]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])

        prev_player_num = prev_obs['active']
        prev_ball_x, prev_ball_y, prev_ball_z = prev_obs['ball']
        prev_player_pos_x, prev_player_pos_y = prev_obs['left_team'][prev_player_num]
        prev_ball_x_relative = prev_ball_x - prev_player_pos_x
        prev_ball_y_relative = prev_ball_y - prev_player_pos_y
        prev_ball_distance = np.linalg.norm([prev_ball_x_relative, prev_ball_y_relative])

        ball_distance_closer_reward = 0
        if prev_ball_distance > ball_distance:
            ball_distance_closer_reward = 1.0
        else:
            ball_distance_closer_reward = -1.0
        #reward = 5.0*win_reward + 5.0*rew + 0.003*ball_position_r + 0.3*yellow_r + 1.0*change_ball_owned_reward + 0.1*ball_not_owned_reward
        reward = 1.0*win_reward + 5.0*rew + 0.003*ball_position_r + 0.3*yellow_r + 1.0*change_ball_owned_reward + 0.003*ball_distance_closer_reward
        
    return reward