import logging
import copy
# from operator import itemgetter
import torch
from crowd_sim.envs.utils.info import *
import time

# import torch.nn.functional as F
from crowd_sim.envs.utils.state import JointState
# from crowd_sim.envs.utils.agent import get_full_state


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, print_failure=False, dynamic_obs=False):
        if dynamic_obs:
            logging.info('Dynamic Obstacle Set TRUE!')
        else:
            logging.info('Dynamic Obstacle Set FALSE!')
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        # check model type
        # print("inside explorer: run k ep")
        # model = self.target_policy        
        # dqn_model = str(type(model)) == "<class 'crowd_nav.policy.lstm_dqn.LstmDQN_t'>"
        # # check policy 
        p = str(type(self.robot.policy))
        orca_dis = p == "<class 'crowd_nav.policy.orca_discrete.ORCA_discrete'>"
        # print(orca_dis)
        # update_action = False
        # if imitation_learning and orca_dis and dqn_model:
        #     update_action = True
        #     logging.debug("update action set TRUE")
        # else:
        #     logging.debug("update action set FALSE")

        for i in range(k):
            if dynamic_obs and phase == 'train':
                ob = self.env.reset1(phase)
            else:
                ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            # old_policies = []
            # print("i, k:", i," ",k)
            # c = 0
            while not done:
                action = self.robot.act(ob)                
                ob, reward, done, info = self.env.step(action)
                # print("state:")
                # print(self.robot.policy.last_state)

                # get state
                last_state = self.robot.policy.last_state
                states.append(last_state)
                # get action
                idx = self.robot.policy.action_space.index(action)
                # print(idx)
                actions.append(idx)
                # action_tensor = torch.zeros(len(self.robot.policy.action_space))
                # action_tensor[idx] = 1
                # actions.append(action_tensor)   
                # get reward
                rewards.append(reward)     
                # get state_           
                next_state = JointState(self.robot.get_full_state(), ob)
                next_state = self.target_policy.transform(next_state)
                # print(next_state.shape)
                next_states.append(next_state)
                # get done
                done = done
                dones.append(not done)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
                

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):                
                # print("collision")
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                # print("timeout")
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')
            # print("done loop",c)
            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set                                    
                    self.dqn_update_memory(states, actions, rewards, next_states, dones, orca_dis)
                    
                    # self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
            
    def run_k_episodes_from_file(self, k, phase, file_name, update_memory=False, imitation_learning=False, episode=None,
                                 print_failure=False):
        self.robot.policy.set_phase(phase)
        t1 = time.time()
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        low_success = 0
        mid_success = 0
        high_success = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            # ob = self.env.reset(phase)
            ob = self.env.reset_from_file(phase, file_name, i)
            # print(i)
            # print(self.env.humans)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                t1 = time.time()
                action = self.robot.act(ob)
                t2 = time.time()
                # print(t2 - t1)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                if i < 300:
                    low_success += 1
                elif i < 700 and i >= 300:
                    mid_success += 1
                else:
                    high_success += 1
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('The number of Collision cases : %d', collision)
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('The number of Timeout cases : %d', timeout)
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
            logging.info('The number of Success cases with humans<5: %d', low_success)
            logging.info('The number of Success cases with humans=5: %d', mid_success)
            logging.info('The number of Success cases with humans>5: %d', high_success)
            # print('The number of Success cases :', success)
            # print('The number of Collision cases :', len(timeout_cases))
        t2 = time.time()
        # if not imitation_learning:
        #     print('rl explorer time:',t2-t1)



    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    # branch
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))

    def dqn_update_memory(self, states, actions, rewards, next_states, dones, transform=True):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        # print("inside explorer: update a2c mem")

        for i, state in enumerate(states):
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            # print(next_state)
            # print(next_state.shape)
            if transform:
                state = self.target_policy.transform(state)            
            action = torch.Tensor([action]).to(self.device)
            reward = torch.Tensor([reward]).to(self.device)        
            # next_state = torch.Tensor([next_state.unsqueeze(0)]).to(self.device)
            done = torch.Tensor([done]).to(self.device)

            item = torch.cat((action, reward, done),0)
            
            self.memory.push((state, item, next_state))

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0