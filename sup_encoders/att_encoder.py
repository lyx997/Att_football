import numpy as np
import torch

def state_to_tensor(state_dict):

    player_state = torch.from_numpy(state_dict["player_state"]).float().unsqueeze(0)
    opp_state = torch.from_numpy(state_dict["opp_state"]).float().unsqueeze(0)
    ball_state = torch.from_numpy(state_dict["ball_state"]).float().unsqueeze(0)
    left_team_state = torch.from_numpy(state_dict["left_team_state"]).float().unsqueeze(0)
    right_team_state = torch.from_numpy(state_dict["right_team_state"]).float().unsqueeze(0)

    state_dict_tensor = {

      "player_state" : player_state,
      "opp_state" : opp_state,
      "ball_state" : ball_state,
      "left_team_state" : left_team_state,
      "right_team_state" : right_team_state,
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
        player_num = obs['active']
        
        player_direction = np.array(obs['left_team_direction'][player_num])
        player_role = obs['left_team_roles'][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs['left_team_tired_factor'][player_num]

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
                      }

        return state_dict

    def _encode_role_onehot(self, role_num):
        role_num = np.array(role_num)
        num = role_num.size
        result = np.zeros((num, 10))
        nums = np.arange(0, num, 1)
        result[nums, role_num] = 1
        return result
