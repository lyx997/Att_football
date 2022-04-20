import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gfootball.env as football_env


env_right = football_env.create_environment(env_name="11_vs_11_competition", representation="raw", stacked=False, logdir='/tmp/football', number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=1,\
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)

obs = env_right.reset()

