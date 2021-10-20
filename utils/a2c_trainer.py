import logging
import os
import copy
import torch
from crowd_sim.envs.utils.info import *
# import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from crowd_sim.envs.utils.state import JointState
import matplotlib.pyplot as plt
import pandas as pd
import pickle

class A2c_Trainer(object):
    def __init__(self, env, robot, device, model, epsilon_start, epsilon_end, epsilon_decay ,gamma=None, checkpoint_interval=None, output_dir=None, evaluation_interval= None):
        self.env = env
        self.robot = robot
        self.device = device
        self.gamma = gamma
        self.model = model
        self.target_model = None
        self.optimizer = None
        self.dynamic_obs = False
        # epsilon
        # self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir

        self.evaluation_interval = evaluation_interval
        # self.case_size = None
        self.k = 1
        self.phase = None


    def set_learning_rate(self, learning_rate):
        # self.model = self.robot.policy.get_model()
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)


    def a2c_run(self, train_episode, phase, episode=None, resume=False, dynamic_obs=False):
        self.phase = phase
        # check evaluation 
        # if episode % self.evaluation_interval == 0:
        #     phase == 'val'
        #     self.case_size = self.env.case_size['val']
        # self.robot.policy.set_phase(phase)
        
        # debug
        # print("inside a2c trainer")
        # self.model = self.robot.policy.get_model()
        # a2c_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c.A2CNet'>"
        # a2c_t_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c_t.A2CNet'>"
        # check policy 
        # p = str(type(self.robot.policy))
        # print(p)
        # print(str(type(self.model)))
        if resume:
            data_path = os.path.join(self.output_dir,'key_data.data')
            if not os.path.exists(data_path):
                logging.error('RL key data does not exist')
            with open(data_path, 'rb') as f:
                # store the data as binary data stream
                key_data = pickle.load(f)
            all_lengths = key_data[0]
            average_lengths = key_data[1]
            all_rewards = key_data[2]
            entropy_term = key_data[3]
            logging.debug('RL key data loaded.')
        else:     
            all_lengths = []
            average_lengths = []
            all_rewards = []
            entropy_term = 0

        if dynamic_obs:
            logging.info('Dynamic Obstacle Set TRUE!')
        else:
            logging.info('Dynamic Obstacle Set FALSE!')

        while episode <= train_episode:
            # set phase
            if self.phase == 'test':
                self.k = self.env.case_size['test']
                logging.debug("FINAL TEST")
            elif episode % self.evaluation_interval == 0 and episode!=0:
                self.phase = 'val'
                # self.case_size = self.env.case_size['val']
                self.k = self.env.case_size['val']
            else:
                self.phase = 'train'
                self.k = 1
            self.robot.policy.set_phase(self.phase)

            # for each episode
            if episode < self.epsilon_decay:
                epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) / self.epsilon_decay * episode
            else:
                epsilon = self.epsilon_end
            self.robot.policy.set_epsilon(epsilon)
            # config records
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
            
            # k episode loop 
            for i in range(self.k):
            
                if dynamic_obs and self.phase == 'train':
                    ob = self.env.reset1(self.phase)
                else:
                    ob = self.env.reset(self.phase)
                # ob = self.env.reset(phase)
                done = False

                rewards = []
                log_probs = []
                values = []
                # print("episode, train_episode:", episode," ",train_episode)
                step = 0
                # ac_loss = 0
                while not done:
                    action = self.robot.act(ob)
                    ob, reward, done, info = self.env.step(action)
                    input = self.robot.policy.last_state.unsqueeze(0)
                    
                    prob, V = self.model(input)
                    value = V.detach().numpy()[0,0]
                    dist = prob.detach().numpy()
                    # print(prob.shape)

                    action_idx = self.robot.policy.action_space.index(action)

                    p = prob.squeeze(0)[action_idx]
                    # print('inside a2c trainer')
                    # print(p)
                    # print(type(p))
                    # x = torch.log(torch.Tensor([1e-9]).squeeze(0))
                    # x = 
                    # print(x)
                    # print(type(x))
                    # temp =torch.from_numpy(np.array(1e-9)).double() 
                    # print(temp)
                    # print(type(temp))
                    # print(torch.DoubleTensor([1e-9]))
                    log_prob = torch.log(p)
                    entropy = -np.sum(np.mean(dist) * np.log(dist))
                    # print(dist)
                    # print(type(prob))
                    # print(prob.shape)
                    # print(V)
                    # print(type(V))
                    # print(V.shape)
                    step += 1

                    # states.append(last_state)
                    rewards.append(reward)
                    values.append(value)
                    log_probs.append(log_prob)
                    entropy_term += entropy
                                    

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
                # print("done loop",step)
                

                if self.phase == 'train':
                    # optimize this episode if in train phase
                    next_state = JointState(self.robot.get_full_state(), ob)
                    next_state = self.robot.policy.transform(next_state)
                    # action = self.robot.act(ob)
                    input = next_state.unsqueeze(0)                
                    _, Qval = self.model(input)
                    Qval = Qval.detach().numpy()[0,0]

                    # compute Q value
                    Qvals = np.zeros_like(values)  # returns an array of given shape and type as given array, with zeros.
                    for t in reversed(range(len(rewards))):
                        Qval = rewards[t] + self.gamma * Qval
                        Qvals[t] = Qval

                    # update actor crtic
                    values = torch.FloatTensor(values)
                    Qvals = torch.FloatTensor(Qvals)
                    # print(log_probs)
                    log_probs = torch.stack(log_probs)
                    print(log_probs)

                    advantage = Qvals - values
                    actor_loss = (-log_probs * advantage).mean()
                    critic_loss = 0.5 * advantage.pow(2).mean()
                    ac_loss = actor_loss + critic_loss #+ 0.001 * entropy_term

                    self.optimizer.zero_grad()
                    ac_loss.backward()

                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)

                    self.optimizer.step()    

                    # get some data
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(step)
                    average_lengths.append(np.mean(all_lengths[-10:]))

                    # old code
                    cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)* reward for t, reward in enumerate(rewards)]))

                # save data for unknow disruptive error, overwrite everytime
                if i==0 and episode!=0 and episode % self.checkpoint_interval == 0:
                    key_data = [all_lengths, average_lengths, all_rewards, entropy_term]
                    data_path = os.path.join(self.output_dir,'key_data.data')
                    with open(data_path, 'wb') as f:
                        # store the data as binary data stream
                        pickle.dump(key_data, f)
                        logging.debug('saved memory to: {}'.format(data_path))
                # debug
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)

                
            # after 1 episode
            success_rate = success / self.k
            collision_rate = collision / self.k
            assert success + collision + timeout == self.k
            avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

            extra_info = '' if episode is None else 'in episode {} '.format(episode)
            logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}, k: {:d}'.
                        format(self.phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                                average(cumulative_rewards), self.k))
            if self.phase == 'train':
                logging.info('ac_loss: {:.2E}, actor_loss: {:.2E}, critic_loss: {:.2E}, entropy: {:.2E}, entropy_term: {:.2E}, log_probS: {:.2E}, adv: {:.2E}'.format(ac_loss.data.item(),actor_loss.data.item(),critic_loss.data.item(),entropy, entropy_term, log_probs.mean().data.item(), advantage.mean().data.item()))
            if self.phase in ['val', 'test']:
                num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
                logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                            too_close / num_step, average(min_dist))
            
            if self.phase != 'test':
                # end of episode loop control operations
                
                # save checkpoints
                # save model
                file_name = 'rl_model_'+str(episode)+'.pth'
                save_path = os.path.join(self.output_dir,file_name)
                if episode != 0 and episode % self.checkpoint_interval == 0:
                    torch.save(self.model.state_dict(), save_path)
                    logging.info('saved model to: {}'.format(file_name) )
                
            episode += 1
        if self.phase != 'test' and episode>=train_episode:
            # save all key data for further analysis of data
            smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean() # sliding window avg of size 10
            smoothed_rewards = [i for i in smoothed_rewards]
            plt.plot(all_rewards)
            plt.plot(smoothed_rewards)
            plt.plot()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.show()

            # plt.plot(all_lengths)
            # plt.plot(average_lengths)
            # plt.xlabel("Episode")
            # plt.ylabel("Episode_length")
            # plt.show()
            logging.info('Done with training.')
        else:
            logging.info('Done with testing.')
       

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0