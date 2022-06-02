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
        #self.fc_match2_situation = nn.Linear(64,64)       

        self.fc_player_situation = nn.Linear(arg_dict['feature_dims']['player_situation'],64)
        #self.fc_player2_situation = nn.Linear(64,64)       

        self.fc_ball_situation = nn.Linear(arg_dict['feature_dims']['ball_situation'],64)
        #self.fc_ball2_situation = nn.Linear(64,64)       

        self.fc_player_state = nn.Linear(arg_dict["feature_dims"]["player_state"],64)
        #self.fc_player2_state = nn.Linear(64,64)       
        
        self.fc_ball_state = nn.Linear(arg_dict["feature_dims"]["ball_state"],64)
        #self.fc_ball2_state = nn.Linear(64,64)
        
        self.fc_left_state = nn.Linear(arg_dict["feature_dims"]["left_team_state"],64)
        #self.fc_left2_state = nn.Linear(64,64)
        self.fc_right_state  = nn.Linear(arg_dict["feature_dims"]["right_team_state"],64)
        #self.fc_right2_state  = nn.Linear(64,64)

        #self.fc_left_state = nn.Linear(arg_dict["feature_dims"]["left_team_state"], 48)
        #self.fc_right_state = nn.Linear(arg_dict["feature_dims"]["left_team_state"], 48)

        #self.conv1d_left = nn.Conv1d(48, 36, 1, stride=1)
        #self.conv1d_right = nn.Conv1d(48, 36, 1, stride=1)

        #self.fc_left2_state = nn.Linear(36, 64)
        #self.fc_right2_state = nn.Linear(36, 64)

        self.fc_att1_attack_ws = nn.Linear(64,64)
        self.fc_att1_attack_as = nn.Linear(128,1)
        self.fc_att2_attack_ws = nn.Linear(64,64)
        self.fc_att2_attack_as = nn.Linear(128,1)

        self.fc_att1_defence_ws = nn.Linear(64,64)
        self.fc_att1_defence_as = nn.Linear(128,1)
        self.fc_att2_defence_ws = nn.Linear(64,64)
        self.fc_att2_defence_as = nn.Linear(128,1)

        self.fc_cat = nn.Linear(64*9,arg_dict["lstm_size"])

        self.norm_player_situation = nn.LayerNorm(64)
        #self.norm_player2_situation = nn.LayerNorm(64)
        self.norm_ball_situation = nn.LayerNorm(64)
        #self.norm_ball2_situation = nn.LayerNorm(64)
        self.norm_match_situation = nn.LayerNorm(64)
        #self.norm_match2_situation = nn.LayerNorm(64)
        
        self.norm_player_state = nn.LayerNorm(64)
        #self.norm_player2_state = nn.LayerNorm(64)
        self.norm_ball_state = nn.LayerNorm(64)
        #self.norm_ball2_state = nn.LayerNorm(64)
        self.norm_left_state = nn.LayerNorm(64)
        #self.norm_left2_state = nn.LayerNorm(64)
        self.norm_right_state = nn.LayerNorm(64)
        #self.norm_right2_state = nn.LayerNorm(64)

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
        #match_sit_embed = F.relu(self.norm_match2_situation(self.fc_match2_situation(match_sit_embed)))
        player_sit_embed = F.relu(self.norm_player_situation(self.fc_player_situation(player_situation)))
        #player_sit_embed = F.relu(self.norm_player2_situation(self.fc_player2_situation(player_sit_embed)))
        ball_sit_embed = F.relu(self.norm_ball_situation(self.fc_ball_situation(ball_situation)))
        #ball_sit_embed = F.relu(self.norm_ball2_situation(self.fc_ball2_situation(ball_sit_embed)))

        #player_state_embed = F.relu(self.norm_player2_state(self.fc_player2_state(F.relu(self.norm_player_state(self.fc_player_state(player_state))))))
        player_state_embed = F.relu(self.norm_player_state(self.fc_player_state(player_state)))
        #ball_state_embed = F.relu(self.norm_ball2_state(self.fc_ball2_state(F.relu(self.norm_ball_state(self.fc_ball_state(ball_state))))))
        ball_state_embed = F.relu(self.norm_ball_state(self.fc_ball_state(ball_state)))
        #left_team_state_embed = F.relu(self.norm_left2_state(self.fc_left2_state(F.relu(self.norm_left_state(self.fc_left_state(left_team_state))))))  # horizon, batch, n, dim
        left_team_state_embed = F.relu(self.norm_left_state(self.fc_left_state(left_team_state)))  # horizon, batch, n, dim
        #right_team_state_embed = F.relu(self.norm_right2_state(self.fc_right2_state(F.relu(self.norm_right_state(self.fc_right_state(right_team_state))))))
        right_team_state_embed = F.relu(self.norm_right_state(self.fc_right_state(right_team_state)))

        [horizon, batch, dim] = player_state_embed.size()
        player_state_embed = player_state_embed.view(horizon*batch, 1, dim)
        [horizon, batch, dim] = ball_state_embed.size()
        ball_state_embed = ball_state_embed.view(horizon*batch, 1, dim)
        [horizon, batch, n_left, dim] = left_team_state_embed.size()
        left_team_state_embed = left_team_state_embed.view(horizon*batch, n_left, dim)
        [horizon, batch, n_right, dim] = right_team_state_embed.size()
        right_team_state_embed = right_team_state_embed.view(horizon*batch, n_right, dim)
 
        # 1 layer attention ----- Right team embed to Player, Left team
        player_right_ws1 = self.fc_att1_attack_ws(player_state_embed)
        player_right_ws1_repeat = torch.repeat_interleave(player_right_ws1, repeats=n_right+1, dim=1)
        right_player_ws1 = self.fc_att1_attack_ws(right_team_state_embed)#(1,11,64)
        right_player_ws1 = torch.cat([player_right_ws1, right_player_ws1], dim=1)
        player_ws1 = torch.cat([player_right_ws1_repeat, right_player_ws1], dim=-1)
        player_att1 = F.leaky_relu(self.fc_att1_attack_as(player_ws1))
        player_right_att1 = player_att1[:,1:,:]
        #player_right_att1_ = F.gumbel_softmax(player_right_att1, dim=1, hard=True) #(1*10,11,1)
        player_right_att1 = F.softmax(player_right_att1, dim=1) #(1*10,11,1)
        #player_att1_ = F.gumbel_softmax(player_att1, dim=1, hard=True) #(1,11,1)
        player_att1 = F.softmax(player_att1, dim=1) #(1,11,1)
        player_right_embed = F.elu(torch.bmm(player_att1.permute(0,2,1), right_player_ws1))
        #player_right_embed = torch.add(player_right_ws1, right_att1_player_embed)
        
        ball_left_ws1 = self.fc_att1_defence_ws(ball_state_embed)
        ball_left_ws1_repeat = torch.repeat_interleave(ball_left_ws1, repeats=n_left+1, dim=1)
        left_ball_ws1 = self.fc_att1_defence_ws(left_team_state_embed)
        left_ball_ws1 = torch.cat([ball_left_ws1, left_ball_ws1], dim=1)
        ball_ws1 = torch.cat([ball_left_ws1_repeat, left_ball_ws1], dim=-1)
        ball_att1 = F.leaky_relu(self.fc_att1_defence_as(ball_ws1))
        ball_left_att1 = ball_att1[:,1:,:]
        #ball_left_att1_ = F.gumbel_softmax(ball_left_att1, dim=1, hard=True) #(1*10,11,1)
        ball_left_att1 = F.softmax(ball_left_att1, dim=1) #(1*10,11,1)
        #ball_att1_ = F.gumbel_softmax(ball_att1, dim=1, hard=True) #(1,11,1)
        #ball_att1_ = F.softmax(ball_att1, dim=1) #(1,11,1)
        ball_att1 = F.softmax(ball_att1, dim=1) #(1,11,1)
        ball_left_embed = F.elu(torch.bmm(ball_att1.permute(0,2,1), left_ball_ws1))
        #ball_left_embed = torch.add(ball_left_ws1, left_att1_ball_embed)

        right_team_embed_repeat = torch.repeat_interleave(right_team_state_embed, repeats=n_left+1, dim=1)
        left_team_embed_repeat = torch.repeat_interleave(left_team_state_embed, repeats=n_right, dim=0)
        left_team_embed_repeat = left_team_embed_repeat.view(horizon*batch, n_left*n_right, -1)

        right_team_left_ws1 = self.fc_att1_defence_ws(right_team_state_embed)
        right_team_left_ws1_repeat = self.fc_att1_defence_ws(right_team_embed_repeat)
        right_team_left_ws1_repeat = right_team_left_ws1_repeat.view(horizon*batch*n_right, n_left+1, -1)

        left_right_ws1 = self.fc_att1_defence_ws(left_team_embed_repeat)
        left_right_ws1 = left_right_ws1.view(horizon*batch*n_right, n_left, -1)
        right_team_left_ws1 = right_team_left_ws1.view(horizon*batch*n_right, 1, -1)
        left_right_ws1 = torch.cat([right_team_left_ws1, left_right_ws1], dim=1)

        right_team_ws1 = torch.cat([right_team_left_ws1_repeat, left_right_ws1], dim=-1)
        right_team_att1 = F.leaky_relu(self.fc_att1_defence_as(right_team_ws1))
        right_left_team_att1 = right_team_att1[:,1:,:]
        #right_left_team_att1_ = F.gumbel_softmax(right_left_team_att1, dim=1, hard=True) #(1*10,11,1)
        right_left_team_att1 = F.softmax(right_left_team_att1, dim=1) #(1*10,11,1)
        #right_team_att1_ = F.gumbel_softmax(right_team_att1, dim=1, hard=True) #(1,11,1)
        right_team_att1 = F.softmax(right_team_att1, dim=1) #(1,11,1)
        #right_team_att1 = F.softmax(right_team_att1, dim=1)

        right_left_embed = F.elu(torch.bmm(right_team_att1.permute(0,2,1), left_right_ws1))
        right_left_embed = right_left_embed.view(horizon*batch, n_right, -1)
        #right_left_embed = torch.add(right_team_left_ws1, left_att1_right_embed)

        left_team_embed_repeat = torch.repeat_interleave(left_team_state_embed, repeats=n_right+1, dim=1)
        right_team_embed_repeat = torch.repeat_interleave(right_team_state_embed, repeats=n_left, dim=0)
        right_team_embed_repeat = right_team_embed_repeat.view(horizon*batch, n_left*n_right, -1)
        
        left_team_right_ws1 = self.fc_att1_attack_ws(left_team_state_embed)
        left_team_right_ws1_repeat = self.fc_att1_attack_ws(left_team_embed_repeat)
        left_team_right_ws1_repeat = left_team_right_ws1_repeat.view(horizon*batch*n_left, n_right+1, -1)#(1*10,11,64)

        right_left_ws1 = self.fc_att1_attack_ws(right_team_embed_repeat)
        right_left_ws1 = right_left_ws1.view(horizon*batch*n_left, n_right, -1)#(1*10,11,64)
        left_team_right_ws1 = left_team_right_ws1.view(horizon*batch*n_left, 1, -1)
        right_left_ws1 = torch.cat([left_team_right_ws1, right_left_ws1], dim=1)

        left_team_ws1 = torch.cat([left_team_right_ws1_repeat, right_left_ws1], dim=-1)
        left_team_att1 = F.leaky_relu(self.fc_att1_attack_as(left_team_ws1))
        left_right_team_att1 = left_team_att1[:,1:,:]
        #left_right_team_att1_ = F.gumbel_softmax(left_right_team_att1, dim=1, hard=True) #(1*10,11,1)
        left_right_team_att1 = F.softmax(left_right_team_att1, dim=1) #(1*10,11,1)
        #left_team_att1_ = F.gumbel_softmax(left_team_att1, dim=1, hard=True) #(1*10,11,1)
        left_team_att1 = F.softmax(left_team_att1, dim=1) #(1*10,11,1)
        #left_team_att1 = F.softmax(left_team_att1, dim=1) #(1*10,11,1)

        left_right_embed = F.elu(torch.bmm(left_team_att1.permute(0,2,1), right_left_ws1))
        left_right_embed = left_right_embed.view(horizon*batch, n_left, -1)#(1,10,64)
        #left_right_embed = torch.add(left_team_right_ws1, right_att1_left_embed)

       
        # 2 layer attention ----- Left team embed to Player
        player_left_ws2 = self.fc_att2_attack_ws(player_right_embed)
        player_left_ws2_repeat = torch.repeat_interleave(player_left_ws2, repeats=n_left, dim=1)
        left_player_ws2 = self.fc_att2_attack_ws(left_right_embed)
        player_ws2 = torch.cat([player_left_ws2_repeat, left_player_ws2], dim=-1)
        player_att2 = F.leaky_relu(self.fc_att2_attack_as(player_ws2))
        #player_att2_ = F.gumbel_softmax(player_att2, dim=1, hard=True)#(1,10,1)
        player_att2 = F.softmax(player_att2, dim=1)#(1,10,1)

        ball_right_ws2 = self.fc_att2_defence_ws(ball_left_embed)
        ball_right_ws2_repeat = torch.repeat_interleave(ball_right_ws2, repeats=n_right, dim=1)
        right_ball_ws2 = self.fc_att2_defence_ws(right_left_embed)
        ball_ws2 = torch.cat([ball_right_ws2_repeat, right_ball_ws2], dim=-1)
        ball_att2 = F.leaky_relu(self.fc_att2_defence_as(ball_ws2))
        #ball_att2_ = F.gumbel_softmax(ball_att2, dim=1, hard=True)
        ball_att2 = F.softmax(ball_att2, dim=1)

        player_right_att1 = player_right_att1.permute(0,2,1)
        ball_left_att1 = ball_left_att1.permute(0,2,1)
        player_att2 = player_att2.permute(0,2,1)
        ball_att2 = ball_att2.permute(0,2,1)
        left_right_team_att1 = left_right_team_att1.view(horizon*batch, n_left, n_right)
        right_left_team_att1 = right_left_team_att1.view(horizon*batch, n_right, n_left)

        left_player_atted_embed = torch.bmm(player_att2, left_team_state_embed).view(horizon, batch, -1)
        right_player_atted_embed = torch.bmm(player_right_att1, right_team_state_embed).view(horizon, batch, -1)
        right_ball_atted_embed = torch.bmm(ball_att2, right_team_state_embed).view(horizon, batch, -1)
        left_ball_atted_embed = torch.bmm(ball_left_att1, left_team_state_embed).view(horizon, batch, -1)

        #right_left_player_att1_ = torch.bmm(player_att2_, left_team_att1_)
        right_left_player_att1 = torch.bmm(player_att2, left_right_team_att1)
        right_left_player_atted_embed = torch.bmm(right_left_player_att1, right_team_state_embed).view(horizon,batch, -1)

        #left_right_ball_att1_ = torch.bmm(ball_att2_, right_team_att1_)
        left_right_ball_att1 = torch.bmm(ball_att2, right_left_team_att1)
        left_right_ball_atted_embed = torch.bmm(left_right_ball_att1, left_team_state_embed).view(horizon, batch, -1)

        
        cat = torch.cat([match_sit_embed, player_sit_embed, ball_sit_embed, 
                    left_player_atted_embed, right_player_atted_embed, right_left_player_atted_embed,
                    right_ball_atted_embed, left_ball_atted_embed, left_right_ball_atted_embed], -1)

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

        return prob, prob_m, v, h_out, []

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
    

