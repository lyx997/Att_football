import numpy as np
import torch

def state_to_tensor(state_dict):

    match_situation = torch.from_numpy(state_dict["match_situation"]).float().unsqueeze(0)
    player_state = torch.from_numpy(state_dict["player_state"]).float().unsqueeze(0)
    opp_state = torch.from_numpy(state_dict["opp_state"]).float().unsqueeze(0)
    ball_state = torch.from_numpy(state_dict["ball_state"]).float().unsqueeze(0)
    left_team_state = torch.from_numpy(state_dict["left_team_state"]).float().unsqueeze(0)
    right_team_state = torch.from_numpy(state_dict["right_team_state"]).float().unsqueeze(0)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0)

    state_dict_tensor = {

      'match_situation':match_situation,
      "player_state" : player_state,
      "opp_state" : opp_state,
      "ball_state" : ball_state,
      "left_team_state" : left_team_state,
      "right_team_state" : right_team_state,
      "avail" : avail,
    }
    return state_dict_tensor

class FeatureEncoder:
    def __init__(self):
        self.active = -1

    def get_feature_dims(self):
        dims = {
            'player_state':18,
            'opp_state':18,
            'ball_state':18,
            'left_team_state':18,
            'right_team_state':18,
        }
        return dims

    def encode(self, obs):
        score = obs['score']
        steps_left = obs['steps_left']
        player_num = obs['active']
        
        player_direction = np.array(obs['left_team_direction'][player_num])
        player_pos_x, player_pos_y = obs['left_team'][player_num]
        player_role = obs['left_team_roles'][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs['left_team_tired_factor'][player_num]

        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])

        ball_role_onehot = np.zeros(10)
        ball_owned_onehot = np.zeros(3)

        obs_right_team = np.array(obs['right_team'])
        right_team_distance_to_ball = np.linalg.norm(obs_right_team - np.array(obs['ball'][:-1]), axis=1, keepdims=True)
        opp_num = int(np.argmin(right_team_distance_to_ball))

        opp_direction = np.array(obs['right_team_direction'][opp_num])
        opp_role = obs['right_team_roles'][opp_num]
        opp_role_onehot = self._encode_role_onehot(opp_role)
        opp_tired = obs['right_team_tired_factor'][opp_num]

        left_ball_owned_onehot = np.zeros((11,3)) 
        right_ball_owned_onehot = np.zeros((11,3)) 
        if obs['ball_owned_team'] == -1:
            left_ball_owned_onehot[:,2] = 1.0
            right_ball_owned_onehot[:,2] = 1.0
            ball_owned_onehot[2] = 1.0
        elif obs['ball_owned_team'] == 0:
            left_ball_owned_onehot[player_num, 0] = 1.0
            right_ball_owned_onehot[:,1] = 1.0
            ball_owned_onehot[0:2] = 1.0
        elif obs['ball_owned_team'] == 1:
            left_ball_owned_onehot[:,1] = 1.0
            right_ball_owned_onehot[opp_num, 0] = 1.0
            ball_owned_onehot[0:2] = 1.0

        ball_state = np.concatenate((obs['ball'][:-1], obs['ball_direction'][:-1]*20, ball_role_onehot, ball_owned_onehot, [0.]))#18
        player_state = np.concatenate((obs['left_team'][player_num], player_direction*100, player_role_onehot[0], left_ball_owned_onehot[player_num], [player_tired]))#18
        opp_state = np.concatenate((obs['right_team'][opp_num], opp_direction*100, opp_role_onehot[0], right_ball_owned_onehot[opp_num], [opp_tired]))#18

        obs_left_team = np.array(obs['left_team'])
        obs_left_team_direction = np.array(obs['left_team_direction'])
        left_role = obs["left_team_roles"]
        left_role_onehot = self._encode_role_onehot(left_role)
        left_team_tired = np.array(obs['left_team_tired_factor']).reshape(-1,1)
        left_team_state = np.concatenate((obs_left_team, obs_left_team_direction*100, left_role_onehot, left_ball_owned_onehot, left_team_tired), axis=1) #18

        obs_right_team = np.array(obs['right_team'])
        obs_right_team_direction = np.array(obs['right_team_direction'])
        right_role = obs["right_team_roles"]
        right_role_onehot = self._encode_role_onehot(right_role)
        right_team_tired = np.array(obs['right_team_tired_factor']).reshape(-1,1)
        right_team_state = np.concatenate((obs_right_team, obs_right_team_direction*100, right_role_onehot, right_ball_owned_onehot, right_team_tired), axis=1) #18                                  

        right_team_distance_to_left, left_team_distance_to_right = [], []
        obs_player = np.array(obs['left_team'][player_num]).reshape(1, -1)
        obs_player_left_team = np.concatenate([obs_player, obs_left_team], axis=0)
        for i in range(11):
            right_team_distance_to_left.append(np.linalg.norm(obs_right_team - obs['left_team'][i], axis=1, keepdims=True))
            left_team_distance_to_right.append(np.linalg.norm(obs_player_left_team - obs['right_team'][i], axis=1, keepdims=True))
        right_team_distance_to_left = np.array(right_team_distance_to_left)
        left_team_distance_to_right = np.array(left_team_distance_to_right)

        label_left_att = np.zeros((1, 11))
        label_right_att = np.zeros((1, 11))

        avail = self._get_avail(obs, ball_distance)
        score_situation = self._get_score(score) 
        steps_situation = self._get_steps(steps_left)
        match_situation = np.concatenate((steps_situation, score_situation)) #13

        state_dict = {
                      'player_state':player_state,
                      'opp_state':opp_state,
                      'ball_state':ball_state,
                      'left_team_state':left_team_state,
                      'right_team_state':right_team_state,
                      'right_team_distance_to_left':right_team_distance_to_left,
                      'left_team_distance_to_right':left_team_distance_to_right,
                      'label_left_att': label_left_att,
                      'label_right_att': label_right_att,
                      'avail' : avail,
                      'match_situation':match_situation,
                      }

        return state_dict, opp_num

    def _get_score(self, score):
        left_score = score[0]
        right_score = score[1]
        score_diff_one_hot = [0,0,0]
        
        if left_score > right_score:
            score_diff_on_hot = [1,0,0]
        elif left_score == right_score:
            score_diff_one_hot = [0,1,0]
        else:
            score_diff_one_hot = [0,0,1]
        return np.array(score_diff_one_hot)

    def _get_steps(self, steps_left):
        steps_length = 3001
        steps_len = 300
        steps_situation = [0,0,0,0,0,0,0,0,0,0]

        index = (steps_length - steps_left) // steps_len
        if index == 10:
            index = 9
        
        steps_situation[index] = 1
        return np.array(steps_situation)

    def _get_avail(self, obs, ball_distance):
        avail = [1,1,1,1,1,1,1,1,1,1,1,1]
        NO_OP, MOVE, LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SPRINT, RELEASE_MOVE, \
                                                      RELEASE_SPRINT, SLIDE, DRIBBLE, RELEASE_DRIBBLE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        
        if obs['ball_owned_team'] == 1: # opponents owning ball
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        elif obs['ball_owned_team'] == -1 and ball_distance > 0.03 and obs['game_mode'] == 0: # Ground ball  and far from me
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        else: # my team owning ball
            avail[SLIDE] = 0
            
        # Dealing with sticky actions
        sticky_actions = obs['sticky_actions']
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0
            
        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0
            
        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0
            
        
        # if too far, no shot
        #ball_x, ball_y, _ = obs['ball']
        #if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
        #    avail[SHOT] = 0
        #elif (0.64 <= ball_x and ball_x<=1.0) and (-0.27<=ball_y and ball_y<=0.27):
        #    avail[HIGH_PASS], avail[LONG_PASS] = 0, 0
            
            
        #if obs['game_mode'] == 2 and ball_x < -0.7:  # Our GoalKick 
        #    avail = [1,0,0,0,0,0,0,0,0,0,0,0]
        #    avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
        #    return np.array(avail)
        
        #elif obs['game_mode'] == 4 and ball_x > 0.9:  # Our CornerKick
        #    avail = [1,0,0,0,0,0,0,0,0,0,0,0]
        #    avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
        #    return np.array(avail)
        
        #elif obs['game_mode'] == 6 and ball_x > 0.6:  # Our PenaltyKick
        #    avail = [1,0,0,0,0,0,0,0,0,0,0,0]
        #    avail[SHOT] = 1
        #    return np.array(avail)

        return np.array(avail)
        
    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if   (-END_X <= ball_x    and ball_x < -PENALTY_X)and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [1.0,0,0,0,0,0]
        elif (-END_X <= ball_x    and ball_x < -MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
            return [0,1.0,0,0,0,0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
            return [0,0,1.0,0,0,0]
        elif (PENALTY_X < ball_x  and ball_x <=END_X)     and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [0,0,0,1.0,0,0]
        elif (MIDDLE_X < ball_x   and ball_x <=END_X)     and (-END_Y < ball_y     and ball_y < END_Y):
            return [0,0,0,0,1.0,0]
        else:
            return [0,0,0,0,0,1.0]
        

    def _encode_role_onehot(self, role_num):
        role_num = np.array(role_num)
        num = role_num.size
        result = np.zeros((num, 10))
        nums = np.arange(0, num, 1)
        result[nums, role_num] = 1
        return result
