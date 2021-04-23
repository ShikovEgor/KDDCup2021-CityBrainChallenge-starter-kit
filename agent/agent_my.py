import pickle
import gym
from pathlib import Path
import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
with open(path + "/gym_cfg.py", "r") as f:
    pass

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN

from utils import PreprocEnv, get_setup

class TestAgent:
    def __init__(self, env):
        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.phase_passablelane = {}

        agent_dict, self.agent_mask, self.edge_index = get_setup(env)
        self.agent_list = agent_dict.keys()
        self.env_preproc = PreprocEnv(self.agent_list)  
        self.n_inter = len(self.agent_list)  
        self.n_agnts = int(self.agent_mask.sum())

        self.net = Net(1, 32, self.max_phase)

        self.action_weight = torch.ones((self.n_agnts, self.max_phase))
        
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    ################################   

    def act(self, obs, mode = 'test'):
        obs_tensor = self.env_preproc.preproc_obs(obs)
        self.action_weight = self.net(obs_tensor, self.edge_index)
        self.curr_phase = torch.multinomial(action, 1)
        
        return self.env_preproc.act_tensor_to_env(self.curr_phase)
        

class Net(torch.nn.Module):
    def __init__(self, node_features, filters, n_phase):
        super(Net, self).__init__()
        self.recurrent = A3TGCN(in_channels = node_features, out_channels = filters, periods = 12)
        self.linear = torch.nn.Linear(filters, 4)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h)


# scenario_dirs = [
#     "test"
# ]

# agent_specs = dict.fromkeys(scenario_dirs, None)
# for i, k in enumerate(scenario_dirs):
#     # initialize an AgentSpec instance with configuration
#     agent_specs[k] = TestAgent()

