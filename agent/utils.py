import torch
import argparse

import CBEngine
import json
import traceback
import logging
import os
import sys
import time
from pathlib import Path
import re
import gym
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

gym.logger.setLevel(gym.logger.ERROR)


N_OBS = 24
N_ACTIONS = 4

class PreprocEnv:
    def __init__(self, a_ids, device = torch.device('cpu')):
        self.a_ids = a_ids
        self.nm_ag_speed =  [str(a) + '_lane_speed' for a in a_ids]
        self.nm_ag_num =  [str(a) + '_lane_vehicle_num' for a in a_ids]
        Nagents = len(a_ids)
        self.Nagents = Nagents
        
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
        # соединяем наблюдения и текущую фазу и отправляем
        return  torch.cat( (self.obs_num, torch.unsqueeze(self.current_phase, 1)), dim=1)
        
    
    def act_tensor_to_env(self, act_tensor): 
        with torch.no_grad():
            self.current_phase = act_tensor
        actions = {}
        act_tensor = act_tensor.cpu() + 1  # приводим к диапазон [1,*]
        for i, a_id in enumerate(self.a_ids):
            actions[a_id] = act_tensor[i].item()
        return actions
    
    def step(self, env, a):
        o, r, d, i = env.step(self.act_tensor_to_env(a))
        # print(i)
        return self.preproc_obs(o), r, d, i

    def reset_state(self):
        self.current_phase = torch.randint(N_ACTIONS, size=(self.Nagents,))


def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if(len(line) == 3 and line[0][0] != '#'):
                configs[line[0]] = line[-1]
    return configs


def process_roadnet(roadnet_file):
    intersections = {}
    roads = {}
    agents = {}
    lane_vehicle_state = {}
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'latitude': float(line[0]),
                        'longitude': float(line[1]),
                        'have_signal': int(line[3]),
                        'end_roads': []
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 +
                                                    i] = list(map(int, line[i * 3:i * 3 + 3]))
                            lane_vehicle_state[road_id * 100 + i] = set()
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5])
                        }
                        intersections[int(line[0])]['end_roads'].append(
                            int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(
                            int(line[-2]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    # agents[int(line[0])] = list(map(int,line[1:]))
                    # 4 in-roads
                    agents[int(line[0])] = intersections[int(
                        line[0])]['end_roads']
    return intersections, roads, agents


def process_delay_index(lines, roads, step):
    vehicles = {}

    for i in range(len(lines)):
        line = lines[i]
        if(line[0] == 'for'):
            vehicle_id = int(line[2])
            now_dict = {
                'distance': float(lines[i + 1][2]),
                'drivable': int(float(lines[i + 2][2])),
                'road': int(float(lines[i + 3][2])),
                'route': list(map(int, list(map(float, lines[i + 4][2:])))),
                'speed': float(lines[i + 5][2]),
                'start_time': float(lines[i + 6][2]),
                't_ff': float(lines[i+7][2])
            }
            vehicles[vehicle_id] = now_dict
            tt = step - now_dict['start_time']
            tt_ff = now_dict['t_ff']
            tt_f_r = 0.0
            current_road_pos = 0
            for pos in range(len(now_dict['route'])):
                if(now_dict['road'] == now_dict['route'][pos]):
                    current_road_pos = pos
            for pos in range(len(now_dict['route'])):
                road_id = now_dict['route'][pos]
                if(pos == current_road_pos):
                    tt_f_r += (roads[road_id]['length'] -
                               now_dict['distance']) / roads[road_id]['speed_limit']
                elif(pos > current_road_pos):
                    tt_f_r += roads[road_id]['length'] / roads[road_id]['speed_limit']

            vehicles[vehicle_id]['delay_index'] = (tt + tt_f_r) / tt_ff

    vehicle_list = list(vehicles.keys())
    delay_index_list = []
    for vehicle_id, dict in vehicles.items():
        # res = max(res, dict['delay_index'])
        delay_index_list.append(dict['delay_index'])

    return delay_index_list, vehicle_list, vehicles


def run_simulation(agent_spec, simulator_cfg_file, gym_cfg):
    logger.info("\n")
    logger.info("*" * 40)

    gym_configs = gym_cfg.gym_cfg().cfg
    simulator_configs = read_config(simulator_cfg_file)
    env = gym.make(
        'CBEngine-v0',
        simulator_cfg_file=simulator_cfg_file,
        thread_num=16,
        gym_dict=gym_configs
    )
    scenario = [
        'test'
    ]

    done = False

    roadnet_path = Path(simulator_configs['road_file_addr'])

    intersections, roads, agents = process_roadnet(roadnet_path)

    observations, infos = env.reset()
    agent_id_list = []
    for k in observations:
        agent_id_list.append(int(k.split('_')[0]))
    agent_id_list = list(set(agent_id_list))
    agent = agent_spec[scenario[0]]
    agent.load_agent_list(agent_id_list)

    # env.set_log(0 )
    # env.set_warning(0 )

    env.set_log(1)
    env.set_warning(0)

    agent.epsilon = 0

    step = 0

    while not done:
        actions = {}
        step += 1
        all_info = {
            'observations': observations,
            'info': infos
        }
        actions = agent.act(all_info)
        # print('step ',  step,' ', actions)
        observations, rewards, dones, infos = env.step(actions)
        for agent_id in agent_id_list:
            if (dones[agent_id]):
                done = True
    time = env.eng.get_average_travel_time()

    # read log file
    log_path = Path(simulator_configs['report_log_addr'])
    result = {}
    vehicle_last_occur = {}
    for dirpath, dirnames, file_names in os.walk(log_path):
        for file_name in [f for f in file_names if f.endswith(".log") and f.startswith('info_step')]:
            with open(log_path / file_name, 'r') as log_file:
                pattern = '[0-9]+'
                step = list(map(int, re.findall(pattern, file_name)))[0]
                if (step >= int(simulator_configs['max_time_epoch'])):
                    continue
                lines = log_file.readlines()
                lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
                result[step] = {}
                # result[step]['vehicle_num'] = int(lines[0][0])

                # process delay index
                # delay_index, vehicle_list = process_delay_index(lines, roads, step)
                delay_index_list, vehicle_list, vehicles = process_delay_index(lines, roads, step)

                result[step]['vehicle_list'] = vehicle_list
                result[step]['delay_index'] = delay_index_list
                result[step]['vehicles'] = vehicles
    steps = list(result.keys())
    steps.sort()
    for step in steps:
        for vehicle in result[step]['vehicles'].keys():
            vehicle_last_occur[vehicle] = result[step]['vehicles'][vehicle]

    delay_index_temp = {}
    for vehicle in vehicle_last_occur.keys():
        res = vehicle_last_occur[vehicle]['delay_index']
        delay_index_temp[vehicle] = res

    # last calc
    vehicle_total_set = set()
    delay_index = []
    for k, v in result.items():
        # vehicle_num.append(v['vehicle_num'])
        vehicle_total_set = vehicle_total_set | set(v['vehicle_list'])
        # delay_index.append(v['delay_index'])
        delay_index += delay_index_list
    if (len(delay_index) > 0):
        d_i = np.mean(delay_index)
    else:
        d_i = -1

    last_d_i = np.mean(list(delay_index_temp.values()))
    delay_index = list(delay_index_temp.values())

    return len(vehicle_total_set), last_d_i


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
