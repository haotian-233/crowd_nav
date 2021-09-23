import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
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
        # print("inside trainer:")
        a2c_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c.A2CNet'>"
        # print(a2c_model)
        a2c_t_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c_t.A2CNet'>"
        # print(a2c_t_model)

        if a2c_model or a2c_t_model:
            # model is lstm_ga3c
            for epoch in range(num_epochs):
                epoch_loss = 0
                for data in self.data_loader:
                    inputs, values = data # values shape [100,1]
                    inputs = Variable(inputs)
                    values = Variable(values)
                    # print("input size:", inputs.shape)
                    self.optimizer.zero_grad()
                    logits, output_values = self.model(inputs) # values shape [100,1]
                    # stopped here
                    probs = F.softmax(logits, dim=1)
                    adv = values - output_values
                    critic_loss = adv.pow(2)

                    # dist = torch.distributions.Categorical
                    m = self.model.distribution(probs)
                    action = m.sample().numpy() # numpy array, len 100
                    action = torch.from_numpy(action)
                    # print(type(action))
                    # print(action)
                    # print(action.shape)

                    exp_v = m.log_prob(action) * adv.detach().squeeze()
                    actor_loss = -exp_v
                    total_loss = (critic_loss + actor_loss).mean()
                    
                    total_loss.backward()
                    self.optimizer.step()

                    epoch_loss += total_loss.data.item()

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
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        # check model type
        a2c_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c.A2CNet'>"
        a2c_t_model = str(type(self.model)) == "<class 'crowd_nav.policy.lstm_ga3c_t.A2CNet'>"
        if a2c_model or a2c_t_model:
            for _ in range(num_batches):       
                inputs, values = next(iter(self.data_loader))
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                logits, output_values = self.model(inputs) # values shape [100,1]
                probs = F.softmax(logits, dim=1)
                adv = values - output_values
                critic_loss = adv.pow(2)
                m = self.model.distribution(probs)
                action = m.sample().numpy() # numpy array, len 100
                action = torch.from_numpy(action)
                exp_v = m.log_prob(action) * adv.detach().squeeze()
                actor_loss = -exp_v
                total_loss = (critic_loss + actor_loss).mean()
                total_loss.backward()
                self.optimizer.step()

                losses += total_loss.data.item()
        else:
            for _ in range(num_batches):       
                inputs, values = next(iter(self.data_loader))
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
