import torch
import argparse

N_OBS = 24
N_ACTIONS = 4

# def get_setup(env):
#     agent_dict = {}
#     agent_mask = []
#     for i, (k,v) in enumerate(env.intersections.items()):
#         agent_mask.append(v['have_signal'])
#         agent_dict[k] = i

#     agent_mask = torch.tensor(agent_mask).to(dtype=torch.bool)
#     edge_index = []
#     for v in env.roads.values():
#         edge_index.append( (agent_dict[v['start_inter']], agent_dict[v['end_inter']] ) )
#     edge_index = torch.tensor(edge_index).to(dtype=torch.long).T
#     return agent_dict, agent_mask, edge_index

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
            r = rwd[aid]
            for flow in r:
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
        # self.obs_speed,     
        return  self.obs_num
    
    def act_tensor_to_env(self, act_tensor): 
        actions = {}
        act_tensor = act_tensor.cpu()+1  # приводим к диапазон [1,*]
        for i, a_id in enumerate(self.a_ids):
            actions[a_id] = act_tensor[i].item()
        return actions
    
    def step(self, env, a):
        
        o, r, d, i = env.step(self.act_tensor_to_env(a))
        print(i)
        return self.preproc_obs(o), r, d, i

def get_args():
    parser = argparse.ArgumentParser(
            prog="evaluation",
            description="1"
        )

    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the reference "
             "data and user submission data.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="The path to the directory where the submission's "
             "scores.txt file will be written to.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--sim_cfg",
        help='The path to the simulator cfg',
        default=None,
        type=str
    )

    # Add more argument for training.

    parser.add_argument('--thread', type=int, default=8, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=4, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=2, help='training episodes')
    parser.add_argument('--save_model', action="store_true", default=False)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument("--save_rate", type=int, default=5,
                        help="save model once every time this many episodes are completed")
    parser.add_argument('--save_dir', type=str, default="model/presslight_1234",
                        help='directory in which model should be saved')
    parser.add_argument('--log_dir', type=str, default="cmd_log/presslight_1234", help='directory in which logs should be saved')

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)

    result = {
            "success": False,
            "error_msg": "",
            "data": {
                "total_served_vehicles": -1,
                "delay_index": -1}
            }
    return parser.parse_args('')
