from torch.utils.data import Dataset

import random
from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))

class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

        # embed a2c memory into it
        self.max_episode_length = 100 # time_limit 25/time_step 0.25
        self.num_episode = self.capacity//self.max_episode_length
        self.a2c_memory = deque(maxlen=self.max_episode_length)
        self.trajectory = []

    # new functions for a2c mem
    def append(self, state, action, reward, policy):
        self.trajectory.append(Transition(state, action, reward, policy))  # Save s_i,a_i, r_i+1, µ(·|s_i)
        # Terminal states are saved with actions as None, so switch to next episode
        if action is None:
            self.a2c_memory.append(self.trajectory)
            self.trajectory = []
        # Samples random trajectory
    def sample(self, maxlen=0):
        # print("inside mem, sample:")
        # print(len(self.a2c_memory))
        mem = self.a2c_memory[random.randrange(len(self.a2c_memory))]
        T = len(mem)
        # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
        if maxlen > 0 and T > maxlen + 1:
            t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
            return mem[t:t + maxlen + 1]
        else:
            return mem

    # Samples batch of trajectories, truncating them to the same length
    def sample_batch(self, batch_size, maxlen=0):
        batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
        minimum_size = min(len(trajectory) for trajectory in batch)
        batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
        return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

    def length(self):
        # Return number of epsiodes saved in memory
        return len(self.a2c_memory)


    # old functions for normal mem
    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()
