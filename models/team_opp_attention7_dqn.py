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

        self.fc_q_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_q_a2 = nn.Linear(164, 19)
        self.norm_q_a1 = nn.LayerNorm(164)
        
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])
        
    def forward(self, state_dict):

        player_situation = state_dict["player_situation"]          
        ball_situation = state_dict["ball_situation"]              
        match_situation = state_dict["match_situation"]              

        player_state = state_dict["player_state"]          
        ball_state = state_dict["ball_state"]              
        left_team_state = state_dict["left_team_state"]
        right_team_state = state_dict["right_team_state"]  
        
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
        out, h_out = self.lstm(cat, h_in)
        
        qa_out = F.relu(self.norm_q_a1(self.fc_q_a1(out)))
        q_a = self.fc_q_a2(qa_out)

        return q_a, h_out, [player_right_att1.squeeze().squeeze(), left_right_att2.squeeze().squeeze(), player_left_att2.squeeze().squeeze()]

   
    

