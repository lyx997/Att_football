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

        self.fc_player_state = nn.Linear(arg_dict["rw_feature_dims"]["player_state"],48)
        self.fc_left_state = nn.Linear(arg_dict["rw_feature_dims"]["player_state"],48)
        self.fc_right_state = nn.Linear(arg_dict["rw_feature_dims"]["player_state"],48)
        
        #multi head attention Q, K, V
        self.fc_q1_offence = nn.Linear(48, 48, bias=False)
        self.fc_v1_offence = nn.Linear(48, 48, bias=False)
        self.fc_k1_offence = nn.Linear(48, 48, bias=False)

        self.fc_q2_offence = nn.Linear(96, 96, bias=False)
        self.fc_k2_offence = nn.Linear(96, 96, bias=False)

        self.norm_state = nn.LayerNorm(48)
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])
        
    def forward(self, state_dict):

        player_state = state_dict["player_state"]          
        left_team_state = state_dict["left_team_state"]
        right_team_state = state_dict["right_team_state"]  
        
        player_state_embed = F.relu(self.norm_state(self.fc_player_state(player_state)))
        left_team_state_embed = F.relu(self.norm_state(self.fc_left_state(left_team_state)))  
        right_team_state_embed = F.relu(self.norm_state(self.fc_right_state(right_team_state)))
        
        [batch, dim] = player_state_embed.size()
        player_state_embed = player_state_embed.reshape(batch, 1, dim)
        #[batch, n_left, dim] = left_team_state_embed.size()
        #left_team_state_embed = left_team_state_embed.reshape(batch, n_left, dim) #(1, 10, 48)
        #[batch, n_right, dim] = right_team_state_embed.size()
        #right_team_state_embed = right_team_state_embed.reshape(batch, n_right, dim) #(1, 11, 48)

        player_q1 = self.fc_q1_offence(player_state_embed) # 1,1,48
        player_v1 = self.fc_v1_offence(player_state_embed) # 1,1,48
        right_team_k1 = self.fc_k1_offence(right_team_state_embed) # 1, 11, 48
        right_team_v1 = self.fc_v1_offence(right_team_state_embed) # 1, 11, 48
        player_att_to_right = torch.bmm(player_q1, right_team_k1.permute(0,2,1)) # 1,1,11
        player_att_to_right_ = F.softmax(player_att_to_right, dim=-1)
        player_att_right_team_embed = torch.cat([player_v1, torch.bmm(player_att_to_right_, right_team_v1)], dim=-1) # 1,1,96

        left_team_q1 = self.fc_q1_offence(left_team_state_embed) # 1,11,48
        left_team_v1 = self.fc_v1_offence(left_team_state_embed) # 1,11,48
        left_att_to_right = torch.bmm(left_team_q1, right_team_k1.permute(0,2,1)) # 1,11,11
        left_att_to_right_ = F.softmax(left_att_to_right, dim=-1)
        left_att_right_team_embed = torch.cat([left_team_v1, torch.bmm(left_att_to_right_, right_team_v1)], dim=-1) # 1,11,96

        player_q2 = self.fc_q2_offence(player_att_right_team_embed) #1,1,96
        left_team_k2 = self.fc_k2_offence(left_att_right_team_embed)#1,11,96

        player_att_to_left = torch.bmm(player_q2, left_team_k2.permute(0,2,1)) #1,1,11
        player_att_to_left_ = F.softmax(player_att_to_left, dim=-1)

        return player_att_to_left_.squeeze(0), left_att_to_right_.squeeze(0)

    def make_batch(self, data):
            # data = [tr1, tr2, ..., tr10] * batch_size
            s_player_batch, s_left_batch, s_right_batch, label_left_att_batch, right_left_dis_batch = [],[],[],[],[]

            for rollout in data:
                for transition in rollout:
                    s = transition
                    s_player_batch.append(s["player_state"])
                    s_left_batch.append(s["left_team_state"])
                    s_right_batch.append(s["right_team_state"])
                    label_left_att_batch.append(s["label_left_att"])
                    right_left_dis_batch.append(s["right_team_distance_to_left"])
            
            s = {
              "player_state": torch.tensor(s_player_batch, dtype=torch.float, device="cpu"),
              "left_team_state": torch.tensor(s_left_batch, dtype=torch.float, device="cpu"),
              "right_team_state": torch.tensor(s_right_batch, dtype=torch.float, device="cpu"),
              "label_player_att": torch.tensor(label_left_att_batch, dtype=torch.float, device="cpu"),
              "label_opp_att": torch.tensor(right_left_dis_batch, dtype=torch.float, device="cpu").squeeze(-1),
            }

            return s


