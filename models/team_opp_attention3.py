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

        self.fc_match_situation = nn.Linear(arg_dict['feature_dims']['match_situation'],128)
        self.fc_match2_situation = nn.Linear(128,128)       

        self.fc_player_situation = nn.Linear(arg_dict['feature_dims']['player_situation'],128)
        self.fc_player2_situation = nn.Linear(128,128)       

        self.fc_ball_situation = nn.Linear(arg_dict['feature_dims']['ball_situation'],128)
        self.fc_ball2_situation = nn.Linear(128,128)       

        self.fc_player_state = nn.Linear(arg_dict["feature_dims"]["player_state"],64)
        self.fc_player2_state = nn.Linear(64,64)       
        
        self.fc_ball_state = nn.Linear(arg_dict["feature_dims"]["ball_state"],64)
        self.fc_ball2_state = nn.Linear(64,64)
        
        self.fc_left_state = nn.Linear(arg_dict["feature_dims"]["left_team_state"],64)
        self.fc_left2_state = nn.Linear(64,64)
        self.fc_right_state  = nn.Linear(arg_dict["feature_dims"]["right_team_state"],64)
        self.fc_right2_state  = nn.Linear(64,64)

        self.fc_ball_q0 = nn.Linear(64,64)
        self.fc_ball_v0 = nn.Linear(64,64)
        
        self.fc_player_k0 = nn.Linear(64,64)
        self.fc_player_q1 = nn.Linear(128,128)
        self.fc_player_q2 = nn.Linear(256,256)
        self.fc_player_v0 = nn.Linear(64,64)
        self.fc_player_v1 = nn.Linear(128,128)
        self.fc_player_v2 = nn.Linear(256,256)

        self.fc_left_team_k0 = nn.Linear(64,64)
        self.fc_left_team_q1 = nn.Linear(128,128)
        self.fc_left_team_k2 = nn.Linear(256,256)
        self.fc_left_team_v0 = nn.Linear(64,64)
        self.fc_left_team_v1 = nn.Linear(128,128)
        self.fc_left_team_v2 = nn.Linear(256,256)

        self.fc_right_team_k0 = nn.Linear(64,64)
        self.fc_right_team_k1 = nn.Linear(128,128)
        self.fc_right_team_v0 = nn.Linear(64,64)
        self.fc_right_team_v1 = nn.Linear(128,128)
        
        self.fc_cat = nn.Linear(128+128+128+512,arg_dict["lstm_size"])


        self.norm_player_situation = nn.LayerNorm(128)
        self.norm_player2_situation = nn.LayerNorm(128)
        self.norm_ball_situation = nn.LayerNorm(128)
        self.norm_ball2_situation = nn.LayerNorm(128)
        self.norm_match_situation = nn.LayerNorm(128)
        self.norm_match2_situation = nn.LayerNorm(128)
        
        self.norm_player_state = nn.LayerNorm(64)
        self.norm_player2_state = nn.LayerNorm(64)
        self.norm_ball_state = nn.LayerNorm(64)
        self.norm_ball2_state = nn.LayerNorm(64)
        self.norm_left_state = nn.LayerNorm(64)
        self.norm_left2_state = nn.LayerNorm(64)
        self.norm_right_state = nn.LayerNorm(64)
        self.norm_right2_state = nn.LayerNorm(64)

        self.norm_player_att0 = nn.LayerNorm(128)
        self.norm_left_att0 = nn.LayerNorm(128)
        self.norm_right_att0 = nn.LayerNorm(128)
        self.norm_player_att1 = nn.LayerNorm(256)
        self.norm_left_att1 = nn.LayerNorm(256)
        self.norm_att2 = nn.LayerNorm(512)

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
        ball_state = state_dict["ball_state"]              
        left_team_state = state_dict["left_team_state"]
        right_team_state = state_dict["right_team_state"]  
        avail = state_dict["avail"]
        
        match_sit_embed = F.relu(self.norm_match_situation(self.fc_match_situation(match_situation)))
        match_sit_embed = F.relu(self.norm_match2_situation(self.fc_match2_situation(match_sit_embed)))
        player_sit_embed = F.relu(self.norm_player_situation(self.fc_player_situation(player_situation)))
        player_sit_embed = F.relu(self.norm_player2_situation(self.fc_player2_situation(player_sit_embed)))
        ball_sit_embed = F.relu(self.norm_ball_situation(self.fc_ball_situation(ball_situation)))
        ball_sit_embed = F.relu(self.norm_ball2_situation(self.fc_ball2_situation(ball_sit_embed)))

        player_state_embed = F.relu(self.norm_player_state(self.fc_player_state(player_state)))
        player_state_embed = self.norm_player2_state(self.fc_player2_state(player_state_embed))
        ball_state_embed   = F.relu(self.norm_ball_state(self.fc_ball_state(ball_state)))
        ball_state_embed   = self.norm_ball2_state(self.fc_ball2_state(ball_state_embed))
        
        left_team_state_embed = F.relu(self.norm_left_state(self.fc_left_state(left_team_state)))  # horizon, batch, n, dim
        left_team_state_embed = F.relu(self.norm_left2_state(self.fc_left2_state(left_team_state_embed)))  # horizon, batch, n, dim
        
        right_team_state_embed = F.relu(self.norm_right_state(self.fc_right_state(right_team_state)))
        right_team_state_embed = F.relu(self.norm_right2_state(self.fc_right2_state(right_team_state_embed)))

        [horizon, batch, dim] = ball_state_embed.size()
        ball_state_embed = ball_state_embed.view(horizon*batch, 1, dim)
        [horizon, batch, dim] = player_state_embed.size()
        player_state_embed = player_state_embed.view(horizon*batch, 1, dim)
        [horizon, batch, n, dim] = left_team_state_embed.size()
        left_team_state_embed = left_team_state_embed.view(horizon*batch, n, dim)
        [horizon, batch, n, dim] = right_team_state_embed.size()
        right_team_state_embed = right_team_state_embed.view(horizon*batch, n, dim)

        # 0 layer attention ----- Ball embed to Player, Left team and Right team 
        ball_q0 = self.fc_ball_q0(ball_state_embed)
        ball_v0 = self.fc_ball_v0(ball_state_embed)
        player_v0 = self.fc_player_v0(player_state_embed)
        left_team_k0 = self.fc_left_team_k0(left_team_state_embed)
        left_team_v0 = self.fc_left_team_v0(left_team_state_embed)
        right_team_k0 = self.fc_right_team_k0(right_team_state_embed)
        right_team_v0 = self.fc_right_team_v0(right_team_state_embed)

        left_att0_ball = torch.bmm(left_team_k0, ball_q0.permute(0,2,1))/8
        right_att0_ball = torch.bmm(right_team_k0, ball_q0.permute(0,2,1))/8
        
        left_att0_ball = F.softmax(left_att0_ball, dim=1)
        right_att0_ball = F.softmax(right_att0_ball, dim=1)

        player_att0_embed = self.norm_player_att0(torch.concat([player_v0, ball_v0], dim=-1))


        ball_att0_left_embed = torch.bmm(left_att0_ball, ball_v0)
        left_att0_embed = self.norm_left_att0(torch.concat([left_team_v0, ball_att0_left_embed], dim=-1))


        ball_att0_right_embed = torch.bmm(right_att0_ball, ball_v0)
        right_att0_embed = self.norm_right_att0(torch.concat([right_team_v0, ball_att0_right_embed], dim=-1))


        # 1 layer attention ----- Right team embed to Player, Left team
        player_q1 = self.fc_player_q1(player_att0_embed)
        player_v1 = self.fc_player_v1(player_att0_embed)
        left_team_q1 = self.fc_left_team_q1(left_att0_embed)
        left_team_v1 = self.fc_left_team_v1(left_att0_embed)
        right_team_k1 = self.fc_right_team_k1(right_att0_embed)
        right_team_v1 = self.fc_right_team_v1(right_att0_embed)

        player_att1_right = torch.bmm(player_q1, right_team_k1.permute(0,2,1))/8
        player_att1_right = F.softmax(player_att1_right, dim=-1)

        right_att1_player_embed = torch.bmm(player_att1_right, right_team_v1)
        #player_att1_embed = torch.concat([player_v1, right_att1_player_embed], dim=-1)
        player_att1_embed = self.norm_player_att1(torch.concat([player_v1, right_att1_player_embed], dim=-1))

        left_team_att1_right = torch.bmm(left_team_q1, right_team_k1.permute(0,2,1))/8
        left_team_att1_right = F.softmax(left_team_att1_right, dim=-1)

        right_att1_left_team_embed = torch.bmm(left_team_att1_right, right_team_v1)
        #left_team_att1_embed = torch.concat([left_team_v1, right_att1_left_team_embed], dim=-1)
        left_team_att1_embed = self.norm_left_att1(torch.concat([left_team_v1, right_att1_left_team_embed], dim=-1))


        # 2 layer attention ----- Left team embed to Player
        player_q2 = self.fc_player_q2(player_att1_embed)
        player_v2 = self.fc_player_v2(player_att1_embed)
        left_team_k2 = self.fc_left_team_k2(left_team_att1_embed)
        left_team_v2 = self.fc_left_team_v2(left_team_att1_embed)

        player_att2_left = torch.bmm(player_q2, left_team_k2.permute(0,2,1))/8
        player_att2_left = F.softmax(player_att2_left, dim=-1)

        left_team_att2_player_embed = torch.bmm(player_att2_left, left_team_v2)
        player_att2_embed = self.norm_att2(torch.concat([player_v2, left_team_att2_player_embed], dim=-1))       
        player_att2_embed = player_att2_embed.view(horizon, batch, -1)
        
        cat = torch.cat([match_sit_embed, player_sit_embed, ball_sit_embed, player_att2_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden"]
        #out, h_out = self.lstm(cat, h_in)
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

        return prob, prob_m, v, h_out

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_match_sit_batch, s_player_sit_batch, s_ball_sit_batch, s_player_batch, s_ball_batch, s_left_batch, s_right_batch, avail_batch = [],[],[],[],[],[],[],[]
        s_match_sit_prime_batch, s_player_sit_prime_batch, s_ball_sit_prime_batch, s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch,  \
                                                  s_right_prime_batch, avail_prime_batch =  [],[],[],[],[],[],[],[]
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []
        
        for rollout in data:
            s_match_sit_lst, s_player_sit_lst, s_ball_sit_lst, s_player_lst, s_ball_lst, s_left_lst, s_right_lst, avail_lst =  [], [], [], [], [], [], [], []
            s_match_sit_prime_lst, s_player_sit_prime_lst, s_ball_sit_prime_lst, s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, \
                                                  s_right_prime_lst, avail_prime_lst =  [], [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, m_lst, r_lst, prob_lst, done_lst, need_move_lst = [], [], [], [], [], []
            
            for transition in rollout:
                s, a, m, r, s_prime, prob, done, need_move = transition

                s_player_sit_lst.append(s["player_situation"])
                s_ball_sit_lst.append(s["ball_situation"])
                s_match_sit_lst.append(s["match_situation"])

                s_player_lst.append(s["player_state"])
                s_ball_lst.append(s["ball_state"])
                s_left_lst.append(s["left_team_state"])
                s_right_lst.append(s["right_team_state"])
                avail_lst.append(s["avail"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)

                s_player_sit_prime_lst.append(s_prime["player_situation"])
                s_ball_sit_prime_lst.append(s_prime["ball_situation"])
                s_match_sit_prime_lst.append(s_prime["match_situation"])
                s_player_prime_lst.append(s_prime["player_state"])
                s_ball_prime_lst.append(s_prime["ball_state"])
                s_left_prime_lst.append(s_prime["left_team_state"])
                s_right_prime_lst.append(s_prime["right_team_state"])
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
            s_left_batch.append(s_left_lst)
            s_right_batch.append(s_right_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])

            s_player_sit_prime_batch.append(s_player_sit_prime_lst)
            s_ball_sit_prime_batch.append(s_ball_sit_prime_lst)
            s_match_sit_prime_batch.append(s_match_sit_prime_lst) 
            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
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
          "left_team_state": torch.tensor(s_left_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_team_state": torch.tensor(s_right_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "avail": torch.tensor(avail_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "hidden" : (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2))
        }

        s_prime = {
          "player_situation": torch.tensor(s_player_sit_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball_situation": torch.tensor(s_ball_sit_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "match_situation": torch.tensor(s_match_sit_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "player_state": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball_state": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "left_team_state": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_team_state": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "hidden" : (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2))
        }

        a,m,r,done_mask,prob,need_move = torch.tensor(a_batch, device=self.device).permute(1,0,2), \
                                         torch.tensor(m_batch, device=self.device).permute(1,0,2), \
                                         torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(prob_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(need_move_batch, dtype=torch.float, device=self.device).permute(1,0,2)

        return s, a, m, r, s_prime, done_mask, prob, need_move
    

