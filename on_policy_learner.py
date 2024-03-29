import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

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

def seperate_write_summary(writer, arg_dict, summary_queue, optimization_step, self_play_board, win_evaluation, score_evaluation, model):
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
    
    mini_window = int(arg_dict['summary_eval_window'])
    if len(win_evaluation)>=mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), optimization_step)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), optimization_step)

        if float(np.mean(win_evaluation)) >= 0.98:
            model_dict = {
                'optimization_step': optimization_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }
            path = arg_dict["log_dir"]+"/model_"+str(optimization_step)+".tar"
            torch.save(model_dict, path)
            print("Win rate 100% Model saved :", path) 
        win_evaluation, score_evaluation = [], []

    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/'+opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), optimization_step)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation

def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, v_loss_lst, \
                  entropy_lst, move_entropy_lst, optimization_step, self_play_board, win_evaluation, score_evaluation, good_pass_evaluation, model):
    win, score, tot_reward, tot_good_pass, game_len = [], [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a,b,c,d,e,opp_num,t1,t2,t3 = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]

        if 'env_evaluation' in arg_dict and opp_num==arg_dict['env_evaluation']:
            win_evaluation.append(a)
            good_pass_evaluation.append(d)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            tot_good_pass.append(d)
            game_len.append(e)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)
            
    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/good_pass', float(np.mean(tot_good_pass)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)
    writer.add_scalar('train/pi_loss', np.mean(pi_loss_lst), n_game)
    writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)
    writer.add_scalar('train/move_entropy', np.mean(move_entropy_lst), n_game)

    if float(np.mean(win)) > 0.98:
            model_dict = {
                'optimization_step': optimization_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }
            path = arg_dict["log_dir"]+"/model_"+str(optimization_step)+".tar"
            torch.save(model_dict, path)
            print("Model saved :", path)

    mini_window = int(arg_dict['summary_game_window'] // 3)
    if len(win_evaluation)>=mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/good_pass_evaluation', float(np.mean(good_pass_evaluation)), n_game)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), n_game)

        if float(np.mean(win_evaluation)) >= 0.98:
            model_dict = {
                'optimization_step': optimization_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }
            path = arg_dict["log_dir"]+"/model_"+str(optimization_step)+".tar"
            torch.save(model_dict, path)
            print("Model saved :", path)

        win_evaluation, score_evaluation, good_pass_evaluation = [], [], []

    selfplay_win_rate = []
    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/'+opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), n_game)
            win_rate = np.mean(self_play_board[opp_num][:mini_window])
            selfplay_win_rate.append(win_rate)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    #if float(np.mean(np.array(selfplay_win_rate))) > 0.87:
    #        model_dict = {
    #            'optimization_step': optimization_step,
    #            'model_state_dict': model.state_dict(),
    #            'optimizer_state_dict': model.optimizer.state_dict(),
    #        }
    #        path = arg_dict["log_dir"]+"/model_"+str(optimization_step)+".tar"
    #        torch.save(model_dict, path)
    #        print("Model saved :", path)

    return win_evaluation, score_evaluation, good_pass_evaluation

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
       
def get_data(queue, arg_dict, model):
    data = []
    for i in range(arg_dict["buffer_size"]):
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]):
            rollout = queue.get()
            mini_batch_np.append(rollout)
        mini_batch = model.make_batch(mini_batch_np)
        data.append(mini_batch)
    return data

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
                                                                 self_play_board, win_evaluation, score_evaluation, model)

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
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
  
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)
    
    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    n_game = 0
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
    self_play_board = {}

    win_evaluation, score_evaluation, good_pass_evaluation = [], [], []
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            last_saved_step = save_model(model, arg_dict, optimization_step, last_saved_step)
            signal_queue.put(1)
            data = get_data(queue, arg_dict, model)
            loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, data)
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]
            print("step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize())
            
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            move_entropy_lst.append(move_entropy)
            center_model.load_state_dict(model.state_dict())
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())
            
            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win_evaluation, score_evaluation, good_pass_evaluation = write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, 
                                                                 v_loss_lst, entropy_lst, move_entropy_lst, optimization_step, 
                                                                 self_play_board, win_evaluation, score_evaluation, good_pass_evaluation, model)
                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
                n_game += arg_dict["summary_game_window"]
                
            _ = signal_queue.get()             
            
        else:
            time.sleep(0.1)
           

def sup_learner(center_model, queue, signal_queue, summary_queue, arg_dict):
    print("Learner process started")
    imported_model = importlib.import_module("sup_models." + arg_dict["rl_model"])
    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    algo = imported_algo.Algo(arg_dict)
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
  
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)
    
    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    n_game = 0
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
    self_play_board = {}

    win_evaluation, score_evaluation, good_pass_evaluation = [], [], []
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            last_saved_step = save_model(model, arg_dict, optimization_step, last_saved_step)
            signal_queue.put(1)
            data = get_data(queue, arg_dict, model)
            loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, data)
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]
            print("step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize())
            
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            move_entropy_lst.append(move_entropy)
            center_model.load_state_dict(model.state_dict())
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())
            
            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win_evaluation, score_evaluation, good_pass_evaluation = write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, 
                                                                 v_loss_lst, entropy_lst, move_entropy_lst, optimization_step, 
                                                                 self_play_board, win_evaluation, score_evaluation, good_pass_evaluation, model)
                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst= [], [], [], [], []
                n_game += arg_dict["summary_game_window"]
                
            _ = signal_queue.get()             
            
        else:
            time.sleep(0.1)