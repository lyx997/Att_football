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

        self.norm_player_situation = nn.LayerNorm(64)
        self.norm_ball_situation = nn.LayerNorm(64)
        self.norm_match_situation = nn.LayerNorm(64)

################################################################
        #multi head attention Q, K, V
        self.fc_q1_attack = nn.Linear(arg_dict["feature_dims"]["player_state"], 64)
        self.fc_q1_defence = nn.Linear(arg_dict["feature_dims"]["player_state"], 64)
        self.fc_q2_attack = nn.Linear(64,64)
        self.fc_q2_defence = nn.Linear(64,64)
        self.fc_v1_attack = nn.Linear(arg_dict["feature_dims"]["player_state"], 64)
        self.fc_v1_defence = nn.Linear(arg_dict["feature_dims"]["player_state"], 64)
        #self.fc_v2 = nn.Linear(64,64)
        self.fc_k1_attack = nn.Linear(arg_dict["feature_dims"]["player_state"], 64)
        self.fc_k1_defence = nn.Linear(arg_dict["feature_dims"]["player_state"], 64)
        self.fc_k2_attack = nn.Linear(64,64)
        self.fc_k2_defence = nn.Linear(64,64)

################################################################

        self.fc_cat = nn.Linear(64*9,arg_dict["lstm_size"])
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["lstm_size"],arg_dict["lstm_size"])

        self.fc_pi_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_a2 = nn.Linear(164, 12)
        self.norm_pi_a1 = nn.LayerNorm(164)
        
        self.fc_pi_m1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_V1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_V1 = nn.LayerNorm(164)
        self.fc_V2 = nn.Linear(164, 1,  bias=False)
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
        player_sit_embed = F.relu(self.norm_player_situation(self.fc_player_situation(player_situation)))
        ball_sit_embed = F.relu(self.norm_ball_situation(self.fc_ball_situation(ball_situation)))

        [horizon, batch, dim] = ball_state.size()
        ball_state = ball_state.reshape(horizon*batch, 1, dim)
        [horizon, batch, dim] = player_state.size()
        player_state = player_state.reshape(horizon*batch, 1, dim)
        [horizon, batch, n_left, dim] = left_team_state.size()
        left_team_state = left_team_state.reshape(horizon*batch, n_left, dim)
        [horizon, batch, n_right, dim] = right_team_state.size()
        right_team_state = right_team_state.reshape(horizon*batch, n_right, dim)

        ##-attack_attention---------------------------------------------------------------------------
        # 1 layer attention ----- Right team embed to Player, Left team
        player_q1 = self.fc_q1_attack(player_state)
        player_k1 = self.fc_k1_attack(player_state)
        player_v1 = self.fc_v1_attack(player_state)#(1,1,64)
        left_team_q1 = self.fc_q1_attack(left_team_state)
        left_team_k1 = self.fc_k1_attack(left_team_state)
        left_team_v1 = self.fc_v1_attack(left_team_state)#(1,10,64)
        right_team_k1 = self.fc_k1_attack(right_team_state)
        right_team_v1 = self.fc_v1_attack(right_team_state)#(1,11,64)

        player_right_team_k1 = torch.cat([player_k1, right_team_k1], dim=1)
        player_right_team_v1 = torch.cat([player_v1, right_team_v1], dim=1)#(1,12,64)

        player_att1_right = torch.bmm(player_q1, player_right_team_k1.permute(0,2,1)) / 8 #64
        player_att1_right = F.softmax(player_att1_right, dim=-1)#(1,1,12)
        player_right_att1 = player_att1_right[:,:,1:]

        player_right_att1_embed = torch.bmm(player_att1_right, player_right_team_v1)#(1,1,64)
        player_right_att1_embed = torch.add(player_right_att1_embed, player_v1)

        left_team_att1_right = torch.bmm(left_team_q1, right_team_k1.permute(0,2,1)) / 8
        left_right_att1 = F.softmax(left_team_att1_right, dim=-1)#(1,10,11)
        left_right_att1_ = F.gumbel_softmax(left_team_att1_right, dim=-1, hard=True)#(1,10,11)
        left_team_self_att1 = torch.diagonal(torch.bmm(left_team_q1, left_team_k1.permute(0,2,1)) / 8, dim1=1, dim2=2).unsqueeze(-1)
        left_self_right_att1 = torch.cat([left_team_self_att1, left_team_att1_right], dim=-1) #(1,10,12)
        left_self_right_att1 = left_self_right_att1.view(horizon*batch*n_left, -1, n_right+1) #(10,1,12)
        left_team_att1_right = F.softmax(left_self_right_att1, dim=-1)#(10,1,12)

        right_team_v1_repeat = torch.repeat_interleave(right_team_v1, repeats=n_left, dim=0)#(10,11,64)
        right_team_v1_repeat = right_team_v1_repeat.view(horizon*batch, n_left, n_right, -1)#(1,10,11,64)
        left_team_v1_ = left_team_v1.view(horizon*batch, n_left, 1, -1)#(1,10,1,64)
        left_team_right_v1 = torch.cat([left_team_v1_, right_team_v1_repeat], dim=2)#(1,10,12,64)
        left_team_right_v1 = left_team_right_v1.view(horizon*batch*n_left, n_right+1, -1)#(10,12,64)
        left_team_right_att1_embed = torch.bmm(left_team_att1_right, left_team_right_v1)#(10,1,64)
        left_team_right_att1_embed = left_team_right_att1_embed.view(horizon*batch, n_left, -1)#(1,10,64)
        left_team_right_att1_embed = torch.add(left_team_right_att1_embed, left_team_v1)


        # 2 layer attention ----- Left team embed to Player
        player_q2 = self.fc_q2_attack(player_right_att1_embed)#(1,1,64)
        left_team_k2 = self.fc_k2_attack(left_team_right_att1_embed)#(1,10,64)
        #left_team_v2 = self.fc_v2(left_team_right_att1_embed)#(1,10,64)

        player_left_team_att2 = torch.bmm(player_q2, left_team_k2.permute(0,2,1)) / 8
        player_left_att2 = F.softmax(player_left_team_att2, dim=-1)#(1,1,10)
        player_left_att2_ = F.gumbel_softmax(player_left_team_att2, dim=-1, hard=True)#(1,1,10)

        left_team_att1_right = left_team_att1_right.permute(1,0,2)
        left_right_att2 = torch.bmm(player_left_att2, left_right_att1)#(1,1,11)
        left_right_att2_ = torch.bmm(player_left_att2_, left_right_att1_)#(1,1,11)

        left_team_att2_player_embed = torch.bmm(player_left_att2, left_team_v1)#(1,1,64)
        left_team_att2_player_embed_ = torch.bmm(player_left_att2_, left_team_v1)#(1,1,64)
        left_team_att2_player_embed = torch.add(left_team_att2_player_embed, left_team_att2_player_embed_)
        left_team_att2_player_embed = left_team_att2_player_embed.view(horizon, batch, -1)

        right_team_att2_left_embed = torch.bmm(left_right_att2, right_team_v1)
        right_team_att2_left_embed_ = torch.bmm(left_right_att2_, right_team_v1)
        right_team_att2_left_embed = torch.add(right_team_att2_left_embed, right_team_att2_left_embed_)
        right_team_att2_left_embed = right_team_att2_left_embed.view(horizon, batch, -1)

        player_right_att1_embed = player_right_att1_embed.view(horizon, batch, -1)
        ##---------------------------------------------------------------------------------
 
        ##-defence_attention---------------------------------------------------------------------------
        # 1 layer attention ----- Right team embed to Player, Left team
        ball_q1 = self.fc_q1_defence(ball_state)
        ball_k1 = self.fc_k1_defence(ball_state)
        ball_v1 = self.fc_v1_defence(ball_state)#(1,1,64)
        right_team_q1 = self.fc_q1_defence(right_team_state)
        right_team_k1 = self.fc_k1_defence(right_team_state)
        right_team_v1 = self.fc_v1_defence(right_team_state)#(1,10,64)
        left_team_k1 = self.fc_k1_defence(left_team_state)
        left_team_v1 = self.fc_v1_defence(left_team_state)#(1,11,64)

        ball_left_team_k1 = torch.cat([ball_k1, left_team_k1], dim=1)
        ball_left_team_v1 = torch.cat([ball_v1, left_team_v1], dim=1)#(1,12,64)

        ball_att1_left = torch.bmm(ball_q1, ball_left_team_k1.permute(0,2,1)) / 8 #64
        ball_att1_left = F.softmax(ball_att1_left, dim=-1)#(1,1,12)
        ball_left_att1 = ball_att1_left[:,:,1:]

        ball_left_att1_embed = torch.bmm(ball_att1_left, ball_left_team_v1)#(1,1,64)
        ball_left_att1_embed = torch.add(ball_left_att1_embed, ball_v1)#(1,1,64)

        right_team_att1_left = torch.bmm(right_team_q1, left_team_k1.permute(0,2,1)) / 8
        right_left_att1 = F.softmax(right_team_att1_left, dim=-1)#(1,10,11)
        right_left_att1_ = F.gumbel_softmax(right_team_att1_left, dim=-1, hard=True)#(1,10,11)
        right_team_self_att1 = torch.diagonal(torch.bmm(right_team_q1, right_team_k1.permute(0,2,1)) / 8, dim1=1, dim2=2).unsqueeze(-1)
        right_self_left_att1 = torch.cat([right_team_self_att1, right_team_att1_left], dim=-1) #(1,10,12)
        right_self_left_att1 = right_self_left_att1.view(horizon*batch*n_right, -1, n_left+1) #(10,1,12)
        right_team_att1_left = F.softmax(right_self_left_att1, dim=-1)#(10,1,12)

        left_team_v1_repeat = torch.repeat_interleave(left_team_v1, repeats=n_right, dim=0)#(10,11,64)
        left_team_v1_repeat = left_team_v1_repeat.view(horizon*batch, n_right, n_left, -1)#(1,10,11,64)
        right_team_v1_ = right_team_v1.view(horizon*batch, n_right, 1, -1)#(1,10,1,64)
        right_team_left_v1 = torch.cat([right_team_v1_, left_team_v1_repeat], dim=2)#(1,10,12,64)
        right_team_left_v1 = right_team_left_v1.view(horizon*batch*n_right, n_left+1, -1)#(10,12,64)
        right_team_left_att1_embed = torch.bmm(right_team_att1_left, right_team_left_v1)#(10,1,64)
        right_team_left_att1_embed = right_team_left_att1_embed.view(horizon*batch, n_right, -1)#(1,10,64)
        right_team_left_att1_embed =torch.add(right_team_left_att1_embed, right_team_v1)


        # 2 layer attention ----- right team embed to ball
        ball_q2 = self.fc_q2_defence(ball_left_att1_embed)#(1,1,64)
        right_team_k2 = self.fc_k2_defence(right_team_left_att1_embed)#(1,10,64)
        #right_team_v2 = self.fc_v2(right_team_left_att1_embed)#(1,10,64)

        ball_right_team_att2 = torch.bmm(ball_q2, right_team_k2.permute(0,2,1)) / 8
        ball_right_att2 = F.softmax(ball_right_team_att2, dim=-1)#(1,1,10)
        ball_right_att2_ = F.gumbel_softmax(ball_right_team_att2, dim=-1, hard=True)#(1,1,10)

        right_team_att1_left = right_team_att1_left.permute(1,0,2)
        right_left_att2 = torch.bmm(ball_right_att2, right_left_att1)#(1,1,11)
        right_left_att2_ = torch.bmm(ball_right_att2_, right_left_att1_)#(1,1,11)

        right_team_att2_ball_embed = torch.bmm(ball_right_att2, right_team_v1)#(1,1,64)
        right_team_att2_ball_embed_ = torch.bmm(ball_right_att2_, right_team_v1)#(1,1,64)
        right_team_att2_ball_embed = torch.add(right_team_att2_ball_embed, right_team_att2_ball_embed_)
        right_team_att2_ball_embed = right_team_att2_ball_embed.view(horizon, batch, -1)

        left_team_att2_right_embed = torch.bmm(right_left_att2, left_team_v1)
        left_team_att2_right_embed_ = torch.bmm(right_left_att2_, left_team_v1)
        left_team_att2_right_embed = torch.add(left_team_att2_right_embed, left_team_att2_right_embed_)
        left_team_att2_right_embed = left_team_att2_right_embed.view(horizon, batch, -1)

        ball_left_att1_embed = ball_left_att1_embed.view(horizon, batch, -1)
        ##---------------------------------------------------------------------------------       
        cat = torch.cat([match_sit_embed, player_sit_embed, ball_sit_embed, player_right_att1_embed, right_team_att2_left_embed, left_team_att2_player_embed, 
                        ball_left_att1_embed, left_team_att2_right_embed, right_team_att2_ball_embed], dim=2)
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

        v = F.relu(self.norm_V1(self.fc_V1(out)))
        v = self.fc_V2(v)

        return prob, prob_m, v, h_out, [player_right_att1.squeeze().squeeze(), left_right_att2.squeeze().squeeze(), player_left_att2.squeeze().squeeze()], \
                [ball_left_att1.squeeze().squeeze(), right_left_att2.squeeze().squeeze(), ball_right_att2.squeeze().squeeze()]

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