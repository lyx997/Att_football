import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
import numpy as np


class Algo():
    def __init__(self, arg_dict, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        #self.lmbda = arg_dict["lmbda"]
        #self.eps_clip = arg_dict["eps_clip"]
        #self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]

    def train(self, eval_model, target_model, data):

        v_loss_lst = []

        for i in range(self.K_epoch):
            for mini_batch in data:
                s, a, r, s_prime, done_mask = mini_batch
                with torch.no_grad():
                    q_a, _, _ = eval_model(s)
                    q_a_prime, _, _ = target_model(s_prime)

                v = q_a.gather(2,a)
                v_prime = torch.max(q_a_prime.detach(), dim=2)[0].unsqueeze(-1)

                td_target = r + self.gamma * v_prime * done_mask

                loss = F.smooth_l1_loss(v, td_target.detach())
                #loss = loss.mean()

                eval_model.optimizer.zero_grad()
                loss.requires_grad = True
                loss.backward()
                nn.utils.clip_grad_norm_(eval_model.parameters(), self.grad_clip)
                eval_model.optimizer.step()

                v_loss_lst.append(loss.item())

        return np.mean(v_loss_lst)
