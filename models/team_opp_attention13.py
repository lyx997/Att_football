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

        self.fc_player_situation = nn.Linear(arg_dict['feature_dims']['player_situation'],48)

        self.fc_ball_situation = nn.Linear(arg_dict['feature_dims']['ball_situation'],48)

        self.fc_left = nn.Linear(arg_dict["feature_dims"]["player_state"],48)
        self.fc_right  = nn.Linear(arg_dict["feature_dims"]["player_state"],48)
        
        self.conv1d_left = nn.Conv1d(48, 36, 1, stride=1)
        self.conv1d_right = nn.Conv1d(48, 36, 1, stride=1)
        self.fc_left2 = nn.Linear(36*10,96)
        self.fc_right2 = nn.Linear(36*11,96)
        
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)

        self.norm_att = nn.LayerNorm(arg_dict['feature_dims']["player_state"])
        
        #multi head attention Q, K, V
        self.fc_q1 = nn.Linear(arg_dict["feature_dims"]["player_state"], 48, bias=False)
        self.fc_v1 = nn.Linear(arg_dict["feature_dims"]["player_state"], 48, bias=False)
        self.fc_k1 = nn.Linear(arg_dict["feature_dims"]["player_state"], 48, bias=False)

        self.fc_q2 = nn.Linear(96, 96, bias=False)
        self.fc_v2 = nn.Linear(96, 96, bias=False)
        self.fc_k2 = nn.Linear(96, 96, bias=False)
        
        self.fc_cat = nn.Linear(48*2+96*2+36,arg_dict["lstm_size"])

        self.norm_situation = nn.LayerNorm(48)
       
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["lstm_size"],arg_dict["lstm_size"])

        self.fc_pi_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_a2 = nn.Linear(164, 12)
        self.norm_pi_a1 = nn.LayerNorm(164)
        
        self.fc_pi_m1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_value1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_value1 = nn.LayerNorm(164)
        self.fc_value2 = nn.Linear(164, 1,  bias=False)
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])
        
    def forward(self, state_dict):

        player_situation = state_dict["player_situation"]          
        ball_situation = state_dict["ball_situation"]              
        #match_situation = state_dict["match_situation"]              

        player_state = state_dict["player_state"]          
        left_team_state = state_dict["left_team_state"]
        ball_state = state_dict["ball_state"]
        right_team_state = state_dict["right_team_state"]  
        avail = state_dict["avail"]
        
        player_sit_embed = F.relu(self.norm_situation(self.fc_player_situation(player_situation)))
        ball_sit_embed = F.relu(self.norm_situation(self.fc_ball_situation(ball_situation)))

        player_state_embed = self.norm_att(player_state)
        left_team_state_embed = self.norm_att(left_team_state)
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim
        right_team_state_embed = self.norm_att(right_team_state)
        right_team_embed = self.norm_right(self.fc_right(right_team_state))  # horizon, batch, n, dim

        [horizon, batch, dim] = player_state_embed.size()
        player_state_embed = player_state_embed.reshape(horizon*batch, 1, dim)
        [horizon, batch, n_left, dim] = left_team_state_embed.size()
        left_team_state_embed = left_team_state_embed.reshape(horizon*batch, n_left, dim) #(1, 10, 48)
        [horizon, batch, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(horizon*batch, n_player, dim).permute(0,2,1)         # horizon * batch, dim1, n
        left_team_embed_ = F.relu(self.conv1d_left(left_team_embed)).permute(0,2,1)                       # horizon * batch, n, dim2
        left_team_embed = left_team_embed_.reshape(horizon*batch, -1).view(horizon,batch,-1)    # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))
        [horizon, batch, n_right, dim] = right_team_state_embed.size()
        right_team_state_embed = right_team_state_embed.reshape(horizon*batch, n_right, dim) #(1, 11, 48)
        [horizon, batch, n_player, dim] = right_team_embed.size()
        right_team_embed = right_team_embed.view(horizon*batch, n_player, dim).permute(0,2,1)         # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0,2,1)                       # horizon * batch, n, dim2
        right_team_embed = right_team_embed.reshape(horizon*batch, -1).view(horizon,batch,-1)    # horizon, batch, n * dim2
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))

        ##-attack_attention---------------------------------------------------------------------------
        # 1 layer attention ----- Right team embed to Player, Left team

        right_team_q1 = self.fc_q1(right_team_state_embed) #(1,11,48)
        right_team_v1 = self.fc_v1(right_team_state_embed) #(1,11,48)
        left_team_k1 = self.fc_k1(left_team_state_embed) #(1,10,48)
        left_team_v1 = self.fc_v1(left_team_state_embed) #(1,10,48)
        player_k1 = self.fc_k1(player_state_embed) #(1,1,48)
        player_v1 = self.fc_v1(player_state_embed) #(1,1,48)
        all_left_team_k1 = torch.cat([player_k1, left_team_k1], dim=1) #(1,11,48)

        right_team_att = torch.bmm(right_team_q1, all_left_team_k1.permute(0,2,1)) #(1,11,11)
        right_team_att_ = F.softmax(right_team_att, dim=-1)
        right_att = right_team_att_.permute(0,2,1) #(1,11,11)
        right_team_att_embed = torch.bmm(right_att, right_team_v1) #(1,11,48)
        right_team_att_player_embed = right_team_att_embed[:,0,:].unsqueeze(1) #(1,1,48)
        right_team_att_left_embed = right_team_att_embed[:,1:,:] #(1,10,48)

        player_right_team_att_embed = torch.cat([player_v1, right_team_att_player_embed], dim=-1) #(1,1,96)
        left_team_right_team_att_embed = torch.cat([left_team_v1, right_team_att_left_embed], dim=-1) #(1,10,96)

        # 2 layer attention ----- Left team embed to Player

        player_q2 = self.fc_q2(player_right_team_att_embed) #(1,1,96)
        player_v2 = self.fc_v2(player_right_team_att_embed)
        left_team_k2 = self.fc_k2(left_team_right_team_att_embed) #(1,10,96)
        left_team_v2 = self.fc_v2(left_team_right_team_att_embed)

        left_team_att = torch.bmm(player_q2, left_team_k2.permute(0,2,1)) #* 100 #(1,1,10)
        left_att_ = F.softmax(left_team_att, dim=-1)
        left_att = F.gumbel_softmax(left_team_att, dim=-1, hard=True)

        left_player_att_embed = torch.bmm(left_att, left_team_embed_) #(1,1,36)
        
        ##---------------------------------------------------------------------------------
        player_right_att_embed = player_v2.view(horizon, batch, -1)
        left_player_att_embed = left_player_att_embed.view(horizon, batch, -1)
 
        cat = torch.cat([player_sit_embed, ball_sit_embed, left_player_att_embed, left_team_embed, right_team_embed], -1)

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

        v = F.relu(self.norm_value1(self.fc_value1(out)))
        v = self.fc_value2(v)

        return prob, prob_m, v, h_out, [left_att_.view(horizon, batch, n_left), right_team_att_.view(horizon, batch, 11, 11)], []#[player_att1_right.view(horizon, batch, 11), left_right_att1.view(horizon, batch, 11, 11), player_left_att2.view(horizon, batch, 11)], \
                #[ball_att1_left.view(horizon, batch, 11), right_left_att1.view(horizon, batch, 11, 11), ball_right_att2.view(horizon, batch, 11)]

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_match_sit_batch, s_player_sit_batch, s_ball_sit_batch, s_player_batch, s_ball_batch, s_left_batch, s_right_batch, avail_batch, right_player_dis_batch, right_left_dis_batch, left_ball_dis_batch, left_right_dis_batch = [],[],[],[],[],[],[],[],[],[],[],[]
        s_match_sit_prime_batch, s_player_sit_prime_batch, s_ball_sit_prime_batch, s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch,  \
                                                  s_right_prime_batch, avail_prime_batch =  [],[],[],[],[],[],[],[]
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []
        
        for rollout in data:
            s_match_sit_lst, s_player_sit_lst, s_ball_sit_lst, s_player_lst, s_ball_lst, s_left_lst, s_right_lst, avail_lst, right_player_dis_lst, right_left_dis_lst, left_ball_dis_lst, left_right_dis_lst =  [], [], [], [], [], [], [], [],[],[],[],[]
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
                right_player_dis_lst.append(s["right_team_distance_to_player"])
                right_left_dis_lst.append(s["right_team_distance_to_left"])
                left_ball_dis_lst.append(s["left_team_distance_to_ball"])
                left_right_dis_lst.append(s["left_team_distance_to_right"])
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
            right_left_dis_batch.append(right_left_dis_lst)
            right_player_dis_batch.append(right_player_dis_lst)
            left_ball_dis_batch.append(left_ball_dis_lst)
            left_right_dis_batch.append(left_right_dis_lst)
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
          "right_player_dis": torch.tensor(right_player_dis_batch, dtype=torch.float, device=self.device).squeeze(-1).permute(1,0,2),
          "left_ball_dis": torch.tensor(left_ball_dis_batch, dtype=torch.float, device=self.device).squeeze(-1).permute(1,0,2),
          "right_left_dis": torch.tensor(right_left_dis_batch, dtype=torch.float, device=self.device).squeeze(-1).permute(1,0,2,3),
          "left_right_dis": torch.tensor(left_right_dis_batch, dtype=torch.float, device=self.device).squeeze(-1).permute(1,0,2,3),
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
          "left_team_state": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_team_state": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
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
    

