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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN
from utils import PreprocEnv
from torch.distributions.categorical import Categorical


class NodeInit(nn.Module):
    def __init__(self, n_agent, n_non_agent, dim, ptype = 'press', sharing = True):
        super().__init__()

        self.n_agent = n_agent
        self.dim = dim
        self.n_non_agent = n_non_agent
        self.phase_emb = nn.Embedding(4, dim)
        self.lin_agg_ph_o = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(inplace=True))
        
        if ptype == 'conv':
            c1 = nn.Conv2d(1, 8, (4, 2),padding=(1,1 ))
            p1 = nn.MaxPool2d(2, padding = 1)
            c2 = nn.Conv2d(8, dim, (3, 2),padding=(1,1 ))
            self.mod  = nn.Sequential(c1, nn.BatchNorm2d(8), nn.ReLU(inplace=True), 
                                      p1, c2, nn.ReLU(inplace=True))
            self.prepr = self.conv_prepr
        else:
            self.mod = nn.Sequential(nn.Linear(8, dim), nn.ReLU(inplace=True))
            self.prepr = self.mean_prepr

        if sharing:
            h_agent = nn.Parameter(torch.Tensor(1,dim))
            h_non_agent = nn.Parameter(torch.Tensor(1,dim))
            nn.init.normal_(h_agent)
            nn.init.normal_(h_non_agent)
            self.h_init = torch.cat((h_agent.expand(n_agent, -1),
                        h_non_agent.expand(n_non_agent, -1)), dim=0)
        else:
            self.h_init = nn.Parameter(torch.Tensor(n_agent+n_non_agent, dim))
            nn.init.normal_(self.h_init)

        with torch.no_grad():
            self.h_feat = torch.zeros((n_agent+n_non_agent, dim))

    def reset(self):
        with torch.no_grad():
            self.h_feat = torch.zeros((self.n_agent+self.n_non_agent,  self.dim))
    
    def conv_prepr(self, x):
        x = x.view((-1,2,4,3)).permute([0,1,3,2]).reshape((-1,6,4))
        x = self.mod(x.unsqueeze(1))
        return x.mean(dim=[2,3])
    
    def mean_prepr(self, x):
        return self.mod(x.view(-1, 8, 3).mean(dim=2))
        
    def forward(self, x):
        h_feat = torch.zeros((self.n_agent+self.n_non_agent,  self.dim))

        obs_features = self.prepr(x[:, :24])
        ph_features = self.phase_emb(x[:, 24].to(dtype=torch.long))
        ag_features = self.lin_agg_ph_o(  torch.cat( ( obs_features,ph_features), dim=1)    )
        h_feat[:22] = ag_features
        return h_feat + self.h_init


class TempGraphNet(torch.nn.Module):
    def __init__(self, edge_index, n_agnts,n_inter,  dim_in, filters, n_phase):
        super(TempGraphNet, self).__init__()
        
        self.recurrent = TGCN(in_channels = dim_in, out_channels = filters)
        # self.recurrent = A3TGCN(in_channels = node_features, out_channels = filters, periods = 12)
        self.linear = torch.nn.Linear(filters, n_phase)
        self.n_agnts = n_agnts
        self.n_inter = n_inter
        self.filters = filters
        self.edge_index = edge_index
        self.reset_state()
    
    def reset_state(self):
        with torch.no_grad():
            self.state = torch.zeros(self.n_inter, self.filters)

    def forward(self, x):
        self.state = self.recurrent(X = x, edge_index = self.edge_index, H = self.state)  
        y = F.relu(self.state)
        y = self.linear(y)
        # return F.softmax(h)
        return y[:self.n_agnts,:]


class CategoricalActor(nn.Module):
    def __init__(self, feat_prepr, net):
        super().__init__()
        self.logits_net = net
        self.feat_prepr = feat_prepr
        
    def _distribution(self, obs):
        x = self.feat_prepr(obs)
        logits = self.logits_net(x)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).mean()
    
    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
class Critic(nn.Module):
    def __init__(self, feat_prepr, net):
        super().__init__()
        self.feat_prepr = feat_prepr
        self.v_net = net

    def forward(self, obs):
        x = self.feat_prepr(obs)
        return self.v_net(x).mean()

class TestAgent:
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.phase_passablelane = {}
        self.ob_length = 24
          
        
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
        
    ################################   

    def get_data_from_env(self, env):
        non_agent_list = []
        for i, (k,v) in enumerate(env.intersections.items()):
            if not v['have_signal']:
                non_agent_list.append(k)

        inter_list = self.agent_list + non_agent_list 
        #сначала светофоры, потом - нерег!!!!!
        agent_dict = {v:k for k,v in enumerate(inter_list)}
        edge_index = []
        for v in env.roads.values():
            edge_index.append( (agent_dict[v['start_inter']], agent_dict[v['end_inter']] ) )
        self.edge_index = torch.tensor(edge_index).to(dtype=torch.long).T

        self.n_inter = len(inter_list)  
        self.n_agnts = len(self.agent_list)
        self.action_weight = torch.ones((self.n_agnts, self.max_phase))
        
        nhid1 = 32
        nhid2 = 32

        self.env_preproc = PreprocEnv(self.agent_list)  
        feat_prepr = NodeInit(self.n_agnts, self.n_inter-self.n_agnts, nhid1, ptype = 'conv', sharing=False) 
        self.pi = CategoricalActor(feat_prepr,
            TempGraphNet(self.edge_index, self.n_agnts, self.n_inter, nhid1, nhid2, self.max_phase))
        self.v  = Critic(feat_prepr,
            TempGraphNet(self.edge_index, self.n_agnts, self.n_inter, nhid1, nhid2, 1))


    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a


    def act(self, obs):
        # используется только при тесте
        obs = self.env_preproc.preproc_obs(obs['observations'])
        a, _, _ = self.step(obs)
        return self.env_preproc.act_tensor_to_env(a)

    def reset(self):
        self.pi.logits_net.reset_state()
        self.v.v_net.reset_state()   
        self.env_preproc.reset_state()     

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

