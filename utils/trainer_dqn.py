import logging
from operator import itemgetter
from git.index import typ
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy



class Trainer(object):
    def __init__(self, model, memory, device, batch_size, gamma=None):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.gamma = gamma
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0

        # add condition, check input model type, branch
        # print("inside trainer, optim epoch:")
        # print(str(type(self.model)))
        dqn_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_dqn.DQNNet'>"

        if dqn_model:
            # model is a2c net
            for epoch in range(num_epochs):
                epoch_loss = 0
                for data in self.data_loader:
                    inputs, items, next_states = data # values shape [100,1]
                    inputs = Variable(inputs)
                    items = Variable(items)
                    next_states = Variable(next_states)
                    # print("input size:", inputs.shape)
                    self.optimizer.zero_grad()

                    action = items[:,0].type(torch.LongTensor).unsqueeze(1)
                    reward = items[:,1].unsqueeze(1)
                    done = items[:,2].unsqueeze(1)

                    Qs = self.model(inputs).gather(1, action) # values shape [bs,81]
                    next_Qs = self.model(next_states)
                    next_Qs = done*next_Qs

                    q_target = reward + self.gamma * torch.max(next_Qs, dim=1)[0].unsqueeze(1)

                    # loss = self.criterion(Qs, q_target.detach())      
                    loss = self.criterion(Qs, q_target.detach())               
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

                    self.optimizer.step()
                    epoch_loss += loss.data.item()
                    # print(loss.data.item())

                # print(epoch_loss)
                # print(len(self.memory))
                average_epoch_loss = epoch_loss / len(self.memory)
                logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)
        else:
            # single return model

            for epoch in range(num_epochs):
                epoch_loss = 0
                for data in self.data_loader:
                    inputs, values = data
                    inputs = Variable(inputs)
                    values = Variable(values)                  
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, values)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.data.item()

                average_epoch_loss = epoch_loss / len(self.memory)
                logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        # check model type
        # a2c_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c.A2CNet'>"
        # a2c_t_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c_t.A2CNet'>"

        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')                     
        # if self.data_loader is None:
        #     self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True) #self.batch_size

        # logging.debug('Experience set size: %d/%d', len(self.memory), self.memory.capacity)        

        losses = 0
        for _ in range(num_batches):
            self.data_loader = DataLoader(self.memory, 1, shuffle=True)
            inputs, items, next_states = next(iter(self.data_loader))

            inputs = Variable(inputs)
            items = Variable(items)
            next_states = Variable(next_states)
            self.optimizer.zero_grad()

            action = items[:,0].type(torch.LongTensor).unsqueeze(1)
            reward = items[:,1].unsqueeze(1)
            done = items[:,2].unsqueeze(1)

            Qs = self.model(inputs).gather(1, action) # values shape [bs,81]
            next_Qs = self.target_model(next_states)
            next_Qs = done*next_Qs

            q_target = reward + self.gamma * torch.max(next_Qs, dim=1)[0].unsqueeze(1)
   
            # loss = self.criterion(Qs, q_target.detach())
            loss = self.criterion(Qs, q_target.detach())   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
            losses += loss.data.item()

        # print('inside optim batch')
        # print(losses)
        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
        
            
            # trajectories = self.memory.sample_batch(self.batch_size, maxlen=self.memory.max_episode_length) # maxlen = 100
            # # repeat for num_batches
            # # for _ in range(num_batches):   
            # policies, Vs, actions, rewards, old_policies = [], [], [], [], []
            # # print(len(trajectories)) # minimum time steps for all trajectories
            # # print(len(trajectories[0])) # num of trajectories
            
            # for i in range(len(trajectories) - 1):
            #     # Unpack first half of transition
            #     state = Variable(torch.cat(tuple(trajectory.state.unsqueeze(0) for trajectory in trajectories[i]), 0))
            #     action = Variable(torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1))
            #     reward = Variable(torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1))
            #     old_policy = Variable(torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0))

            #     # Calculate policy and values
            #     ## convert state to proper dim, replaced by unsqueeze cmd above
            #     # state = state.view(num_batches,-1,self.model.joint_input_dim)
            #     probs, V  = self.model(state)
                
            #     # debug shape
            #     # print(probs.shape)
            #     # print(V.shape)
            #     # print(action.shape)
            #     # print(reward.shape)
            #     # print(old_policy.shape)

            #     # Save outputs for offline training
            #     [arr.append(el) for arr, el in zip((policies, Vs, actions, rewards, old_policies), (probs, V, action, reward, old_policy))]

            #     # Unpack second half of transition
            #     next_state = torch.cat(tuple(trajectory.state.unsqueeze(0) for trajectory in trajectories[i + 1]), 0)
            #     done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1)

            #     # debug shape
            #     # print(next_state.shape)
            #     # print(done.shape)

            # # Do forward pass for all transitions
            # _, Qret = self.model(next_state)
            # # Qret = 0 for terminal s, V(s_i; θ) otherwise
            # Qret = ((1 - done) * Qret).detach()

            # # perform optimization calculations
            # policy_loss, value_loss = 0, 0
            # for i in reversed(range(len(rewards))):
            #     # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
            #     rho = policies[i].detach() / old_policies[i]
                
            #     # Qret ← r_i + γQret
            #     Qret = rewards[i] + self.gamma * Qret
            #     # Advantage A ← Qret - V(s_i; θ)
            #     A = Qret - Vs[i]

            #     # Log policy log(π(a_i|s_i; θ))
            #     log_prob = policies[i].gather(1, actions[i]).log()
            #     # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
            #     single_step_policy_loss = -(rho.gather(1, actions[i]).clamp(max=10) * log_prob * A.detach()).mean(0)  # Average over batch

            #     # Off-policy bias correction
            #     # omitted for now
                
            #     # Policy update dθ ← dθ + ∂θ/∂θ∙g
            #     policy_loss += single_step_policy_loss

            #     # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
            #     policy_loss -= 0.0001 * -(policies[i].log() * policies[i]).sum(1).mean(0)  # Sum over probabilities, average over batch; 0.0001 is entropy weight β

            #     # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
            #     # Q = Qs[i].gather(1, actions[i])
            #     value_loss += ( A ** 2 / 2).mean(0)  # Least squares loss

            #     # print(policy_loss)
            #     # # print(policy_loss.shape)
            #     # print(value_loss)
            #     # print("end")

            # loss = policy_loss + value_loss
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

            # losses += loss.data.item()

            # average_loss = losses

        
