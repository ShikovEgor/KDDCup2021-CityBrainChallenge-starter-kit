import torch
N_OBS = 24
N_ACTIONS = 4

def get_setup(env):
    agent_dict = {}
    agent_mask = []
    for i, (k,v) in enumerate(env.intersections.items()):
        agent_dict[k] = i
        agent_mask.append(v['have_signal'])

    agent_mask = torch.tensor(agent_mask).to(dtype=torch.bool)
    edge_index = []
    for v in env.roads.values():
        edge_index.append( (agent_dict[v['start_inter']], agent_dict[v['end_inter']] ) )
    edge_index = torch.tensor(edge_index).to(dtype=torch.long).T
    return agent_dict, agent_mask, edge_index

class PreprocEnv:
    def __init__(self, a_ids, device = torch.device('cpu')):
        self.a_ids = a_ids
        self.nm_ag_speed =  [str(a) + '_lane_speed' for a in a_ids]
        self.nm_ag_num =  [str(a) + '_lane_vehicle_num' for a in a_ids]
        Nagents = len(a_ids)
        
        self.res_flow = torch.zeros((Nagents, N_OBS))
        self.obs_speed = torch.zeros((Nagents, N_OBS))
        self.obs_num = torch.zeros((Nagents, N_OBS))
        
        self.device = device
        
    @torch.no_grad()    
    def preproc_rwd(self, rwd):
        for iagent, aid in enumerate(self.a_ids):
            for iflow in range(N_OBS):
                flow = rwd[aid][iflow]
                if type(flow) is tuple: 
                    self.res_flow[iagent, iflow] = flow[1]-flow[0]
                else:
                    self.res_flow[iagent, iflow] = 0
        return self.res_flow
    
    @torch.no_grad()
    def preproc_obs(self, obs):
        for iagent,aid in enumerate(self.a_ids):
            self.obs_speed[iagent] = torch.tensor(obs[self.nm_ag_speed[iagent]][1:])
            self.obs_num[iagent] = torch.tensor(obs[self.nm_ag_num[iagent]][1:])
            self.obs_num[self.obs_num==-1] = 0
        return self.obs_speed * self.obs_num
    
    def act_tensor_to_env(self, act_tensor): 
        actions = {}
        act_tensor = act_tensor.cpu()
        for i, a_id in enumerate(self.a_ids):
            actions[a_id] = act_tensor[i].item()
        return actions
    