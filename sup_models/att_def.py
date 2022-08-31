import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(Model, self).__init__()
        self.device=None
        if device:
            self.device = device

        self.arg_dict = arg_dict

        self.fc_ball_state = nn.Linear(arg_dict["def_feature_dims"]["player_state"],48)
        self.fc_left_state = nn.Linear(arg_dict["def_feature_dims"]["player_state"],48)
        self.fc_right_state = nn.Linear(arg_dict["def_feature_dims"]["player_state"],48)
        
        #multi head attention Q, K, V
        self.fc_q1_defence = nn.Linear(48, 48, bias=False)
        self.fc_v1_defence = nn.Linear(48, 48, bias=False)
        self.fc_k1_defence = nn.Linear(48, 48, bias=False)

        self.fc_q2_defence = nn.Linear(96, 96, bias=False)
        self.fc_k2_defence = nn.Linear(96, 96, bias=False)

        self.norm_state = nn.LayerNorm(48)
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])
        
    def forward(self, state_dict):

        ball_state = state_dict["ball_state"]          
        right_team_state = state_dict["right_team_state"]
        left_team_state = state_dict["left_team_state"]  
        
        ball_state_embed = F.relu(self.norm_state(self.fc_ball_state(ball_state)))
        right_team_state_embed = F.relu(self.norm_state(self.fc_right_state(right_team_state)))  
        left_team_state_embed = F.relu(self.norm_state(self.fc_left_state(left_team_state)))
        
        [batch, dim] = ball_state_embed.size()
        ball_state_embed = ball_state_embed.reshape(batch, 1, dim)
        #[horizon, batch, n_right, dim] = right_team_state_embed.size()
        #right_team_state_embed = right_team_state_embed.reshape(horizon*batch, n_right, dim) #(1, 10, 48)
        #[horizon, batch, n_left, dim] = left_team_state_embed.size()
        #left_team_state_embed = left_team_state_embed.reshape(horizon*batch, n_left, dim) #(1, 11, 48)

        ball_q1 = self.fc_q1_defence(ball_state_embed) # 1,1,48
        ball_v1 = self.fc_v1_defence(ball_state_embed) # 1,1,48
        left_team_k1 = self.fc_k1_defence(left_team_state_embed) # 1, 11, 48
        left_team_v1 = self.fc_v1_defence(left_team_state_embed) # 1, 11, 48
        ball_att_to_left = torch.bmm(ball_q1, left_team_k1.permute(0,2,1)) # 1,1,11
        ball_att_to_left_ = F.softmax(ball_att_to_left, dim=-1)
        ball_att_left_team_embed = torch.cat([ball_v1, torch.bmm(ball_att_to_left_, left_team_v1)], dim=-1) # 1,1,96

        right_team_q1 = self.fc_q1_defence(right_team_state_embed) # 1,11,48
        right_team_v1 = self.fc_v1_defence(right_team_state_embed) # 1,11,48
        right_att_to_left = torch.bmm(right_team_q1, left_team_k1.permute(0,2,1)) # 1,11,11
        right_att_to_left_ = F.softmax(right_att_to_left, dim=-1)
        right_att_left_team_embed = torch.cat([right_team_v1, torch.bmm(right_att_to_left_, left_team_v1)], dim=-1) # 1,11,96

        ball_q2 = self.fc_q2_defence(ball_att_left_team_embed) #1,1,96
        right_team_k2 = self.fc_k2_defence(right_att_left_team_embed)#1,11,96

        ball_att_to_right = torch.bmm(ball_q2, right_team_k2.permute(0,2,1)) #1,1,11
        ball_att_to_right_ = F.softmax(ball_att_to_right, dim=-1)

        return ball_att_to_right_.squeeze(0), right_att_to_left_.squeeze(0)

    def make_batch(self, data):
            # data = [tr1, tr2, ..., tr10] * batch_size
            s_ball_batch, s_left_batch, s_right_batch, label_right_att_batch, left_right_dis_batch = [],[],[],[],[]
            for transition in data:

                s = transition

                s_ball_batch.append(s["ball_state"])
                s_left_batch.append(s["left_team_state"])
                s_right_batch.append(s["right_team_state"])
                label_right_att_batch.append(s["label_right_att"])
                left_right_dis_batch.append(s["left_team_distance_to_right"])


            s = {
              "ball_state": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device),
              "left_team_state": torch.tensor(s_left_batch, dtype=torch.float, device=self.device),
              "right_team_state": torch.tensor(s_right_batch, dtype=torch.float, device=self.device),
              "label_player_att": torch.tensor(label_right_att_batch, dtype=torch.float, device=self.device),
              "label_opp_att": torch.tensor(left_right_dis_batch, dtype=torch.float, device=self.device).squeeze(-1),
            }

            return s
