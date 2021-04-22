import CBEngine
import gym
import agent.gym_cfg as gym_cfg
simulator_cfg_file = './cfg/simulator.cfg'
mx_step = 3600
gym_cfg_instance = gym_cfg.gym_cfg()

# from learner import Learner
# from agent.agent import TestAgent
from agent.agent import TestAgent

env = gym.make(
    'CBEngine-v0',
    simulator_cfg_file=simulator_cfg_file,
    thread_num=1,
    gym_dict=gym_cfg_instance.cfg
)


def main():
    ag = TestAgent(env)

    # learner = Learner()

    # if args.use_cuda:
    #     learner.cuda()
    observations, infos = env.reset()

    done = False
    # simulation
    step = 0
    while not done:
        print(len(observations.keys()))
        print(observations)
        actions = ag.act(observations)
        observations, rewards, dones, infos = env.step(actions)

        # ag.env_preproc.preproc_rwd(rewards)

        # learner.train_batch(observations, rewards)
        for agent_id in ag.agent_list:
            if(dones[agent_id]):
                done = True



if __name__ == "__main__":
    main()

