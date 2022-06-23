import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter
from collections import namedtuple, deque

def make_batch(data, device):
    # data = [tr1, tr2, ..., tr10] * batch_size
    s_match_sit_batch, s_player_sit_batch, s_ball_sit_batch, s_player_batch, s_ball_batch, s_left_batch, s_right_batch = [],[],[],[],[],[],[]
    s_match_sit_prime_batch, s_player_sit_prime_batch, s_ball_sit_prime_batch, s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch,  \
                                              s_right_prime_batch =  [],[],[],[],[],[],[]
    h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
    a_batch, r_batch, done_batch = [], [], []
    
    for rollout in data:
        s_match_sit_lst, s_player_sit_lst, s_ball_sit_lst, s_player_lst, s_ball_lst, s_left_lst, s_right_lst =  [], [], [], [], [], [], []
        s_match_sit_prime_lst, s_player_sit_prime_lst, s_ball_sit_prime_lst, s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, \
                                              s_right_prime_lst =  [], [], [], [], [], [], []
        h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
        a_lst, r_lst, done_lst = [], [], []
        
        for transition in rollout:
            s, a, r, s_prime, done = transition
            s_player_sit_lst.append(s["player_situation"])
            s_ball_sit_lst.append(s["ball_situation"])
            s_match_sit_lst.append(s["match_situation"])
            s_player_lst.append(s["player_state"])
            s_ball_lst.append(s["ball_state"])
            s_left_lst.append(s["left_team_state"])
            s_right_lst.append(s["right_team_state"])
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
            h1_out, h2_out = s_prime["hidden"]
            h1_out_lst.append(h1_out)
            h2_out_lst.append(h2_out)
            a_lst.append([a])
            r_lst.append([r])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s_player_sit_batch.append(s_player_sit_lst)
        s_ball_sit_batch.append(s_ball_sit_lst)
        s_match_sit_batch.append(s_match_sit_lst)
        s_player_batch.append(s_player_lst)
        s_ball_batch.append(s_ball_lst)
        s_left_batch.append(s_left_lst)
        s_right_batch.append(s_right_lst)
        h1_in_batch.append(h1_in_lst[0])
        h2_in_batch.append(h2_in_lst[0])
        s_player_sit_prime_batch.append(s_player_sit_prime_lst)
        s_ball_sit_prime_batch.append(s_ball_sit_prime_lst)
        s_match_sit_prime_batch.append(s_match_sit_prime_lst) 
        s_player_prime_batch.append(s_player_prime_lst)
        s_ball_prime_batch.append(s_ball_prime_lst)
        s_left_prime_batch.append(s_left_prime_lst)
        s_right_prime_batch.append(s_right_prime_lst)
        h1_out_batch.append(h1_out_lst[0])
        h2_out_batch.append(h2_out_lst[0])
        a_batch.append(a_lst)
        r_batch.append(r_lst)
        done_batch.append(done_lst)
    
    s = {
      "player_situation": torch.tensor(s_player_sit_batch, dtype=torch.float, device=device).permute(1,0,2),
      "ball_situation": torch.tensor(s_ball_sit_batch, dtype=torch.float, device=device).permute(1,0,2),
      "match_situation": torch.tensor(s_match_sit_batch, dtype=torch.float, device=device).permute(1,0,2),
      "player_state": torch.tensor(s_player_batch, dtype=torch.float, device=device).permute(1,0,2),
      "ball_state": torch.tensor(s_ball_batch, dtype=torch.float, device=device).permute(1,0,2),
      "left_team_state": torch.tensor(s_left_batch, dtype=torch.float, device=device).permute(1,0,2,3),
      "right_team_state": torch.tensor(s_right_batch, dtype=torch.float, device=device).permute(1,0,2,3),
      "hidden" : (torch.tensor(h1_in_batch, dtype=torch.float, device=device).squeeze(1).permute(1,0,2), 
                  torch.tensor(h2_in_batch, dtype=torch.float, device=device).squeeze(1).permute(1,0,2))
    }
    s_prime = {
      "player_situation": torch.tensor(s_player_sit_prime_batch, dtype=torch.float, device=device).permute(1,0,2),
      "ball_situation": torch.tensor(s_ball_sit_prime_batch, dtype=torch.float, device=device).permute(1,0,2),
      "match_situation": torch.tensor(s_match_sit_prime_batch, dtype=torch.float, device=device).permute(1,0,2),
      "player_state": torch.tensor(s_player_prime_batch, dtype=torch.float, device=device).permute(1,0,2),
      "ball_state": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=device).permute(1,0,2),
      "left_team_state": torch.tensor(s_left_prime_batch, dtype=torch.float, device=device).permute(1,0,2,3),
      "right_team_state": torch.tensor(s_right_prime_batch, dtype=torch.float, device=device).permute(1,0,2,3),
      "hidden" : (torch.tensor(h1_out_batch, dtype=torch.float, device=device).squeeze(1).permute(1,0,2), 
                  torch.tensor(h2_out_batch, dtype=torch.float, device=device).squeeze(1).permute(1,0,2))
    }
    a,r,done_mask = torch.tensor(a_batch, device=device).permute(1,0,2), \
                                     torch.tensor(r_batch, dtype=torch.float, device=device).permute(1,0,2), \
                                     torch.tensor(done_batch, dtype=torch.float, device=device).permute(1,0,2)
    return s, a, r, s_prime, done_mask


class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=int(3e5))
    
    def push(self, rollout):
        # append a Transition
        self.memory.append(rollout)
    
    def sample(self, arg_dict, device):
        # sample and return a batch of transitions
        data = []
        for i in range(arg_dict["buffer_size"]):
            transitions_np = random.sample(self.memory, k=arg_dict["batch_size"])
            transitions = make_batch(transitions_np, device)
            data.append(transitions)
        return data
    
    def len(self):
        return len(self.memory)

def seperate_write_loss(i, writer, optimization_step, loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst):

    if i == 0:
        writer.add_scalar('train/att_loss', np.mean(loss_lst), optimization_step)
        writer.add_scalar('train/att_pi_loss', np.mean(pi_loss_lst), optimization_step)
        writer.add_scalar('train/att_v_loss', np.mean(v_loss_lst), optimization_step)
        writer.add_scalar('train/att_entropy', np.mean(entropy_lst), optimization_step)
        writer.add_scalar('train/att_move_entropy', np.mean(move_entropy_lst), optimization_step)

    else:
        writer.add_scalar('train/def_loss', np.mean(loss_lst), optimization_step)
        writer.add_scalar('train/def_pi_loss', np.mean(pi_loss_lst), optimization_step)
        writer.add_scalar('train/def_v_loss', np.mean(v_loss_lst), optimization_step)
        writer.add_scalar('train/def_entropy', np.mean(entropy_lst), optimization_step)
        writer.add_scalar('train/def_move_entropy', np.mean(move_entropy_lst), optimization_step)

def seperate_write_summary(writer, arg_dict, summary_queue, optimization_step, self_play_board, win_evaluation, score_evaluation):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a,b,c,d,opp_num,t1,t2,t3 = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]

        if 'env_evaluation' in arg_dict and opp_num==arg_dict['env_evaluation']:
            win_evaluation.append(a)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            game_len.append(d)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)

    writer.add_scalar('game/win_rate', float(np.mean(win)), optimization_step)
    writer.add_scalar('game/score', float(np.mean(score)), optimization_step)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), optimization_step)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), optimization_step)
    writer.add_scalar('train/step', float(optimization_step), optimization_step)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), optimization_step)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), optimization_step)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), optimization_step)
    
    mini_window = int(arg_dict['summary_game_window'])
    if len(win_evaluation)>=mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), optimization_step)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), optimization_step)
        win_evaluation, score_evaluation = [], []
    
    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/'+opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), optimization_step)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation

def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, optimization_step, self_play_board, win_evaluation, score_evaluation):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a,b,c,d,opp_num,t1,t2,t3 = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]

        if 'env_evaluation' in arg_dict and opp_num==arg_dict['env_evaluation']:
            win_evaluation.append(a)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            game_len.append(d)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)
            
    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)

    mini_window = int(arg_dict['summary_game_window'])
    if len(win_evaluation)>=mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), n_game)
        win_evaluation, score_evaluation = [], []
    
    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/'+opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), n_game)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation

def seperate_save_model(i, model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        if i == 0:
            path = arg_dict["log_dir_att"]+"/model_att_"+str(optimization_step)+".tar"
        else:
            path = arg_dict["log_dir_def"]+"/model_def_"+str(optimization_step)+".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step

def save_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir"]+"/model_"+str(optimization_step)+".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return optimization_step
    else:
        return last_saved_step
       
def get_data(queue, arg_dict, replay_buffer):
    for i in range(arg_dict["batch_size"]*arg_dict["buffer_size"]):
        rollout = queue.get()
        replay_buffer.push(rollout)

def seperate_learner(i, center_model, queue, signal_queue, summary_queue, arg_dict, writer):
    print("Learner process started")
    if i==0:
        imported_model = importlib.import_module("models." + arg_dict["model_att"])
    else:
        imported_model = importlib.import_module("models." + arg_dict["model_def"])

    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    algo = imported_algo.Algo(arg_dict)
    
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)
    
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
    self_play_board = {}

    win_evaluation, score_evaluation = [], []
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            last_saved_step = seperate_save_model(i, model, arg_dict, optimization_step, last_saved_step)
            
            signal_queue.put(1)
            data = get_data(queue, arg_dict, model)
            loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, data)
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]

            if i == 0:
                print("Attack model: step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize())
            else:
                print("Defence model: step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize())
            
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            move_entropy_lst.append(move_entropy)
            center_model.load_state_dict(model.state_dict())
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())
            
            if summary_queue.qsize() > arg_dict["summary_game_window"] and i == 0:
                win_evaluation, score_evaluation = seperate_write_summary(writer, arg_dict, summary_queue, optimization_step, 
                                                                 self_play_board, win_evaluation, score_evaluation)

            if len(loss_lst) >= 10:
                seperate_write_loss(i, writer, optimization_step, loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst)
                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
                
            _ = signal_queue.get()             
            
        else:
            time.sleep(0.1)


def learner(center_model, queue, signal_queue, summary_queue, arg_dict):
    print("Learner process started")
    imported_model = importlib.import_module("models." + arg_dict["model"])
    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    algo = imported_algo.Algo(arg_dict)
    eval_model = imported_model.Model(arg_dict, device)
    eval_model.load_state_dict(center_model.state_dict())
    eval_model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    target_model = imported_model.Model(arg_dict, device)
    target_model.load_state_dict(center_model.state_dict())
    target_model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    
    for state in eval_model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    eval_model.to(device)
    target_model.to(device)
    
    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    n_game = 0
    loss_lst = []
    self_play_board = {}
    replay_buffer = ReplayBuffer()
    episode = 0

    win_evaluation, score_evaluation = [], []
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]:
            last_saved_step = save_model(eval_model, arg_dict, optimization_step, last_saved_step)
            get_data(queue, arg_dict, replay_buffer)

            signal_queue.put(1)
            train_data = replay_buffer.sample(arg_dict, device)
            loss = algo.train(eval_model, target_model, train_data)
            episode += 1
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]

            print("step :", optimization_step, "loss", loss, "data_q", \
                queue.qsize(), "summary_q", summary_queue.qsize(), "buffer_size", replay_buffer.len())

            loss_lst.append(loss)
            center_model.load_state_dict(eval_model.state_dict())
            if episode % arg_dict["target_update_step"] == 0:
                target_model.load_state_dict(eval_model.state_dict())

            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())

            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win_evaluation, score_evaluation = write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, optimization_step, 
                                                                 self_play_board, win_evaluation, score_evaluation)
                loss_lst = []
                n_game += arg_dict["summary_game_window"]

            _ = signal_queue.get()             

        else:
            time.sleep(0.1) 