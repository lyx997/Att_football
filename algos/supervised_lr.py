import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
import numpy as np

def shuffle(data, batch_size):
    batch = {}
    shuffle_index = torch.randint(0, data["left_team_state"].shape[0], (batch_size,))
    for key, value in data.items():
        batch[key] = value[shuffle_index, :].cuda()
    return batch


class Algo():
    def __init__(self, arg_dict, device=None):
        self.K_epoch = arg_dict["k_epoch"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.attention_coef = arg_dict["attention_coef"]
        self.grad_clip = arg_dict["grad_clip"]
        self.batch_size = arg_dict["batch_size"]

    def train(self, model, data):
        tot_loss_lst = []
        att_entropy_lst = []
        att_loss_lst = []

        s = shuffle(data, self.batch_size)
        for i in range(self.K_epoch):

                player_att, _ = model(s)

                label_player_att = s["label_player_att"]
                player_att_log = - torch.log(player_att+1e-8)
                player_att_entropy = torch.diagonal(torch.bmm(player_att, player_att_log.permute(0,2,1)), dim1=1, dim2=2)

                
                #att_loss = F.smooth_l1_loss(player_att, label_player_att)
                att_loss = F.mse_loss(player_att, label_player_att)
                att_entropy_loss = -1*self.entropy_coef*player_att_entropy
                loss = att_loss + att_entropy_loss.mean() 
                #loss = loss.mean()

                model.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                model.optimizer.step()

                tot_loss_lst.append(loss.item())
                att_loss_lst.append(att_loss.item())
                att_entropy_lst.append(player_att_entropy.mean().item())
               
        return np.mean(tot_loss_lst), np.mean(att_loss_lst), np.mean(att_entropy_lst)
