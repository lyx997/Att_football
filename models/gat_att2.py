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

        self.fc_match_situation = nn.Linear(arg_dict['feature_dims']['match_situation'],64)

        self.fc_player_situation = nn.Linear(arg_dict['feature_dims']['player_situation'],64)

        self.fc_ball_situation = nn.Linear(arg_dict['feature_dims']['ball_situation'],64)

        self.fc_all_team_state = nn.Linear(arg_dict["feature_dims"]["player_state"],48)
        
        self.fc_att_attack_ws = nn.Linear(48,48)
        self.fc_att_attack_as = nn.Linear(96,1)

        self.fc_cat = nn.Linear(64*3+48*4,arg_dict["lstm_size"])

        self.norm_player_situation = nn.LayerNorm(64)
        self.norm_ball_situation = nn.LayerNorm(64)
        self.norm_match_situation = nn.LayerNorm(64)
        
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["lstm_size"],arg_dict["lstm_size"])

        self.fc_pi_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_a2 = nn.Linear(164, 12)
        self.norm_pi_a1 = nn.LayerNorm(164)
        
        self.fc_pi_m1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_v1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_v1 = nn.LayerNorm(164)
        self.fc_v2 = nn.Linear(164, 1,  bias=False)
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])
        
    def forward(self, state_dict):

        player_situation = state_dict["player_situation"]          
        ball_situation = state_dict["ball_situation"]              
        match_situation = state_dict["match_situation"]              

        player_state = state_dict["player_state"]          
        all_team_state = state_dict["all_team_state"] 
        avail = state_dict["avail"]
        
        match_sit_embed = F.relu(self.norm_match_situation(self.fc_match_situation(match_situation)))
        player_sit_embed = F.relu(self.norm_player_situation(self.fc_player_situation(player_situation)))
        ball_sit_embed = F.relu(self.norm_ball_situation(self.fc_ball_situation(ball_situation)))

        player_state_embed = F.relu(self.fc_all_team_state(player_state))
        all_team_state_embed = F.relu(self.fc_all_team_state(all_team_state))

        [horizon, batch, dim] = player_state_embed.size()
        player_state_embed = player_state_embed.view(horizon*batch, 1, dim)
        [horizon, batch, n_all, dim] = all_team_state_embed.size()
        all_team_state_embed = all_team_state_embed.view(horizon*batch, n_all, dim)
 
        player_ws = self.fc_att_attack_ws(player_state_embed)
        player_ws_repeat = torch.repeat_interleave(player_ws, repeats=n_all, dim=1)
        all_team_ws = self.fc_att_attack_ws(all_team_state_embed)#(1,22,48)
        player_ws = torch.cat([player_ws_repeat, all_team_ws], dim=-1)
        player_att = F.leaky_relu(self.fc_att_attack_as(player_ws))
        player_att = F.softmax(player_att, dim=1) #(1,22,1)
        player_att_embed = F.elu(torch.bmm(player_att.permute(0,2,1), all_team_ws)).view(horizon, batch, -1)

        player_sort3_att_idx = player_att.sort(dim=1)[1][:,:3,0] #(1,3,1)

        all_team_onehot_1 = torch.zeros((horizon*batch, n_all, 1), device=self.device) #(1,22,1)
        all_team_onehot_2 = torch.zeros((horizon*batch, n_all, 1), device=self.device) #(1,22,1)
        all_team_onehot_3 = torch.zeros((horizon*batch, n_all, 1), device=self.device) #(1,22,1)

        for i, idx in enumerate(player_sort3_att_idx):
            all_team_onehot_1[i, idx[0], 0] = 1
            all_team_onehot_2[i, idx[1], 0] = 1
            all_team_onehot_3[i, idx[2], 0] = 1


        all_team_att1_embed = torch.bmm(all_team_onehot_1.permute(0,2,1), all_team_state_embed).view(horizon, batch, -1)
        all_team_att2_embed = torch.bmm(all_team_onehot_2.permute(0,2,1), all_team_state_embed).view(horizon, batch, -1)
        all_team_att3_embed = torch.bmm(all_team_onehot_3.permute(0,2,1), all_team_state_embed).view(horizon, batch, -1)

        cat = torch.cat([match_sit_embed, player_sit_embed, ball_sit_embed, player_att_embed, all_team_att1_embed, all_team_att2_embed, all_team_att3_embed], -1)

        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)

        a_out = F.relu(self.norm_pi_a1(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail-1)*1e7
        prob = F.softmax(logit, dim=2)
        
        prob_m = F.relu(self.norm_pi_m1(self.fc_pi_m1(out)))
        prob_m = self.fc_pi_m2(prob_m)
        prob_m = F.softmax(prob_m, dim=2)

        v = F.relu(self.norm_v1(self.fc_v1(out)))
        v = self.fc_v2(v)

        player_sort3_att_idx = player_sort3_att_idx.squeeze(0)

        return prob, prob_m, v, h_out, player_sort3_att_idx
    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_match_sit_batch, s_player_sit_batch, s_ball_sit_batch, s_player_batch, s_ball_batch, s_team_batch, avail_batch = [],[],[],[],[],[],[]
        s_match_sit_prime_batch, s_player_sit_prime_batch, s_ball_sit_prime_batch, s_player_prime_batch, s_ball_prime_batch, s_team_prime_batch, avail_prime_batch =  [],[],[],[],[],[],[]
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []
        
        for rollout in data:
            s_match_sit_lst, s_player_sit_lst, s_ball_sit_lst, s_player_lst, s_ball_lst, s_team_lst, avail_lst =  [], [], [], [], [], [], []
            s_match_sit_prime_lst, s_player_sit_prime_lst, s_ball_sit_prime_lst, s_player_prime_lst, s_ball_prime_lst, s_team_prime_lst, avail_prime_lst =  [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, m_lst, r_lst, prob_lst, done_lst, need_move_lst = [], [], [], [], [], []
            
            for transition in rollout:
                s, a, m, r, s_prime, prob, done, need_move = transition

                s_player_sit_lst.append(s["player_situation"])
                s_ball_sit_lst.append(s["ball_situation"])
                s_match_sit_lst.append(s["match_situation"])

                s_player_lst.append(s["player_state"])
                s_ball_lst.append(s["ball_state"])
                s_team_lst.append(s["all_team_state"])
                avail_lst.append(s["avail"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)

                s_player_sit_prime_lst.append(s_prime["player_situation"])
                s_ball_sit_prime_lst.append(s_prime["ball_situation"])
                s_match_sit_prime_lst.append(s_prime["match_situation"])
                s_player_prime_lst.append(s_prime["player_state"])
                s_ball_prime_lst.append(s_prime["ball_state"])
                s_team_prime_lst.append(s_prime["all_team_state"])
                avail_prime_lst.append(s_prime["avail"])
                h1_out, h2_out = s_prime["hidden"]
                h1_out_lst.append(h1_out)
                h2_out_lst.append(h2_out)

                a_lst.append([a])
                m_lst.append([m])
                r_lst.append([r])
                prob_lst.append([prob])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                need_move_lst.append([need_move]),
                
            s_player_sit_batch.append(s_player_sit_lst)
            s_ball_sit_batch.append(s_ball_sit_lst)
            s_match_sit_batch.append(s_match_sit_lst)
            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_team_batch.append(s_team_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])

            s_player_sit_prime_batch.append(s_player_sit_prime_lst)
            s_ball_sit_prime_batch.append(s_ball_sit_prime_lst)
            s_match_sit_prime_batch.append(s_match_sit_prime_lst) 
            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_team_prime_batch.append(s_team_prime_lst)
            avail_prime_batch.append(avail_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            a_batch.append(a_lst)
            m_batch.append(m_lst)
            r_batch.append(r_lst)
            prob_batch.append(prob_lst)
            done_batch.append(done_lst)
            need_move_batch.append(need_move_lst)
        

        s = {
          "player_situation": torch.tensor(s_player_sit_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball_situation": torch.tensor(s_ball_sit_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "match_situation": torch.tensor(s_match_sit_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "player_state": torch.tensor(s_player_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball_state": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "all_team_state": torch.tensor(s_team_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "avail": torch.tensor(avail_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "hidden" : (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2)),
          "left_repeat": torch.tensor(np.ones(10)*11, dtype=torch.long, device=self.device),
          "right_repeat": torch.tensor(np.ones(11)*10, dtype=torch.long, device=self.device),
        }
        

        s_prime = {
          "player_situation": torch.tensor(s_player_sit_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball_situation": torch.tensor(s_ball_sit_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "match_situation": torch.tensor(s_match_sit_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "player_state": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball_state": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "all_team_state": torch.tensor(s_team_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "hidden" : (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2)),
          "left_repeat": torch.tensor(np.ones(10)*11, dtype=torch.long, device=self.device),
          "right_repeat": torch.tensor(np.ones(11)*10, dtype=torch.long, device=self.device),
        }

        a,m,r,done_mask,prob,need_move = torch.tensor(a_batch, device=self.device).permute(1,0,2), \
                                         torch.tensor(m_batch, device=self.device).permute(1,0,2), \
                                         torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(prob_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(need_move_batch, dtype=torch.float, device=self.device).permute(1,0,2)

        return s, a, m, r, s_prime, done_mask, prob, need_move
      