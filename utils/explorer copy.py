import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
import time

import torch.nn.functional as F
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
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
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
        model = self.robot.policy.get_model()

        a2c_model = str(type(model)) == "<class 'crowd_nav.policy.lstm_ga3c.A2CNet'>"
        a2c_t_model = str(type(model)) == "<class 'crowd_nav.policy.lstm_ga3c_t.A2CNet'>"

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            old_policies = []
            # print("i, k:", i," ",k)
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                # print("state:")
                # print(self.robot.policy.last_state)
                # print("state end")

                # get action idex
                last_state = self.robot.policy.last_state
                states.append(last_state)
                rewards.append(reward)
                # action_tensor = torch.zeros(len(self.robot.policy.action_space))
                # action_tensor[idx] = 1
                # print(idx)
                # print(action_tensor)
                # print(action_tensor.shape)
                
                
                # compute next state
                if a2c_model or a2c_t_model:
                    idx = self.robot.policy.action_space.index(action)
                    actions.append(idx)                   
                    next_state = JointState(self.robot.get_full_state(), ob)
                    next_state = self.robot.policy.transform(next_state)
                # print(next_state)
                # print(next_state.shape) # [human num 5,13]

                # print(actions)
                
                # print(rewards)
                # print(states)

                if (a2c_model or a2c_t_model) and (phase == "train") :
                    input = self.robot.policy.last_state.unsqueeze(0)
                    # print(input)
                    # print(input.shape)
                    probs, V = model(input) # values shape [100,1]
                    # print(logits)
                    # print(logits.shape)
                    # probs = F.softmax(logits.detach(), dim=1)
                    old_policies.append(probs)
                    

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
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
                    
                    if a2c_model or a2c_t_model:
                        self.a2c_update_memory(states, actions, rewards, old_policies, next_state ,imitation_learning)
                    else:
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

            # debug
            # print("inside explorer, update mem")
            # print(state)
            # print(state.shape) #[human_num 5, joint_state_dim 13] tensor
            # print(value)
            # print(value.shape) #[1] tensor
            self.memory.push((state, value))

    def a2c_update_memory(self, states, actions, rewards, old_policies, terminal_state ,imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        # print("inside explorer: update a2c mem")

        for i, state in enumerate(states):
            reward = rewards[i]
            action = actions[i]
            policy = old_policies[i]

            self.memory.append(state, action, reward, policy)
        
        self.memory.append(terminal_state, None, None, None)

        
        # print(self.memory.length())
            # ++++++ old code block below +++++
            # VALUE UPDATE
            # if imitation_learning:
            #     # define the value of states in IL as cumulative discounted rewards, which is the same in RL
            #     state = self.target_policy.transform(state)
            #     # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
            #     value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
            #                  * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            # else:
            #     if i == len(states) - 1:
            #         # terminal state
            #         value = reward
            #     else:
            #         next_state = states[i + 1]
            #         gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
            #         # branch
            #         value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0))[1].data.item()
            # value = torch.Tensor([value]).to(self.device)

            # # # transform state of different human_num into fixed-size tensor
            # # if len(state.size()) == 1:
            # #     human_num = 1
            # #     feature_size = state.size()[0]
            # # else:
            # #     human_num, feature_size = state.size()
            # # if human_num != 5:
            # #     padding = torch.zeros((5 - human_num, feature_size))
            # #     state = torch.cat([state, padding])

            # self.memory.push((state, value))

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0