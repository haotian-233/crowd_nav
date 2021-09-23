'''
For experiment purpose, not used in a2c implementation
'''


"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""
import argparse
import configparser
import torch
import torch.nn as nn
from my_utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
os.environ["OMP_NUM_THREADS"] = "1"

from crowd_sim.envs.utils.robot import Robot
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

# config info
# mlp_dims = [256, 256]


class Net(CADRL):
    def __init__(self, action_dim, lstm_hidden_dim = 50):
        super().__init__()

        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_input_dim = self.self_state_dim + self.human_state_dim

        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # policy net
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.policy_net = nn.Sequential(
          nn.Linear(self.self_state_dim + lstm_hidden_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, action_dim)
        )
        self.policy_net.apply(init_weights)

        # value net
        self.value_net = nn.Sequential(
          nn.Linear(self.self_state_dim + lstm_hidden_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 256),
          nn.ReLU(),
          nn.Linear(256, 1)
        )
        self.value_net.apply(init_weights)

        # lstm 
        self.lstm = nn.LSTM(self.human_state_dim, lstm_hidden_dim, batch_first=True)

        # set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, state):

        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        values = self.value_net(joint_state)
        logits = self.policy_net(joint_state)

        return logits, values

    def choose_action(self, state):
        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
#         self.eval()
#         logits, _ = self.forward(s)
#         prob = F.softmax(logits, dim=1).data
#         m = self.distribution(prob)
#         return m.sample().numpy()[0]

#     def loss_func(self, s, a, v_t):
#         self.train()
#         logits, values = self.forward(s)
#         td = v_t - values
#         c_loss = td.pow(2)
        
#         probs = F.softmax(logits, dim=1)
#         m = self.distribution(probs)
#         exp_v = m.log_prob(a) * td.detach().squeeze()
#         a_loss = -exp_v
#         total_loss = (c_loss + a_loss).mean()
#         return total_loss


# class Worker(mp.Process):
#     def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
#         super(Worker, self).__init__()
#         self.name = 'w%02i' % name
#         self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
#         self.gnet, self.opt = gnet, opt
#         self.lnet = Net(N_S, N_A)           # local network
#         self.env = gym.make('CartPole-v0').unwrapped

#     def run(self):
#         total_step = 1
#         while self.g_ep.value < MAX_EP:
#             s = self.env.reset()
#             buffer_s, buffer_a, buffer_r = [], [], []
#             ep_r = 0.
#             while True:
#                 if self.name == 'w00':
#                     self.env.render()
#                 a = self.lnet.choose_action(v_wrap(s[None, :]))
#                 s_, r, done, _ = self.env.step(a)
#                 if done: r = -1
#                 ep_r += r
#                 buffer_a.append(a)
#                 buffer_s.append(s)
#                 buffer_r.append(r)

#                 if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
#                     # sync
#                     push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
#                     buffer_s, buffer_a, buffer_r = [], [], []

#                     if done:  # done and print information
#                         record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
#                         break
#                 s = s_
#                 total_step += 1
#         self.res_queue.put(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    args = parser.parse_args()

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    
    # N_S = env.observation_space.shape[0] -> None
    # N_A = env.action_space.n -> None



    print("Done!")
    # gnet = Net(N_S, N_A)        # global network
    # gnet.share_memory()         # share the global parameters in multiprocessing
    # opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    # global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # # parallel training
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    # [w.start() for w in workers]
    # res = []                    # record episode reward to plot
    # while True:
    #     r = res_queue.get()
    #     if r is not None:
    #         res.append(r)
    #     else:
    #         break
    # [w.join() for w in workers]

    # import matplotlib.pyplot as plt
    # plt.plot(res)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()
