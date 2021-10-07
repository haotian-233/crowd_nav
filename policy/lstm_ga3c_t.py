import torch
import torch.nn as nn
import numpy as np
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from my_utils import process_state

from crowd_sim.envs.utils.action import ActionRot, ActionXY
import torch.nn.functional as F

class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()
        '''
        input_dim = self.self_state_dim + self.human_state_dim = 6+7
        global_state -> lstm_hidden_dim = 50
        mlp(56, mlp_dims)
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        '''
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        # human_state = state[:, :, self.self_state_dim:]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(mlp1_output, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class A2CNet(nn.Module):
    def __init__(self, action_dim, lstm_hidden_dim = 50):
        super().__init__()
        self.device = 'cpu'
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
          nn.Linear(self.self_state_dim + lstm_hidden_dim, 150),
          nn.ReLU(),
          nn.Linear(150, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, action_dim)
        )
        self.policy_net.apply(init_weights)

        # value net
        self.value_net = nn.Sequential(
          nn.Linear(self.self_state_dim + lstm_hidden_dim, 150),
          nn.ReLU(),
          nn.Linear(150, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 1)
        )
        self.value_net.apply(init_weights)

        # lstm 
        self.lstm = nn.LSTM(self.human_state_dim, lstm_hidden_dim, batch_first=True)

        # set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, state):
        '''
        state shape: [batch_size, human_number (0 is self), input_dim = joint_state_dim + 0]
        '''
        # print("inside lstm_ga3c_t")
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        # begin est collision time
        state_temp = torch.zeros(size, dtype=torch.float, device=self.device)
        for i in range(size[0]):
            self_state_temp = state[i, 0, :self.self_state_dim]
            humans_state_temp = state[i, :, :]  # self.self_state_dim
            state_t = process_state(self_state_temp, humans_state_temp)
            state_t = torch.tensor(state_t).to(self.device)
            state_temp[i] = state_t

        human_state = state_temp[:, :, self.self_state_dim:]
        # end 
        
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)

        # print(human_state.shape)
        # print("proper shape:")
        # print(state[:,1:,-self.human_state_dim:].shape)
        output, (hn, cn) = self.lstm(human_state, (h0, c0))
        # output, (hn, cn) = self.lstm(state[:,1:,-self.human_state_dim:], (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        values = self.value_net(joint_state)
        logits = self.policy_net(joint_state)


        # add new output Q, change critic net to predict Q value instead
        probs = F.softmax(logits, dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
        # V = (Q * probs).sum(1, keepdim=True) # V is expectation of Q under π

        return probs, values

class LstmGA3C_t(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-GA3C'
        self.with_interaction_module = None
        self.interaction_module_dims = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_ga3c', 'mlp2_dims').split(', ')]
        global_state_dim = config.getint('lstm_ga3c', 'global_state_dim')
        self.with_om = config.getboolean('lstm_ga3c', 'with_om')
        with_interaction_module = config.getboolean('lstm_ga3c', 'with_interaction_module')

        # set model
        self.model = A2CNet(81,lstm_hidden_dim=50)
        print("class LstmGA3C_t: a2c_t net loaded.")
        # if with_interaction_module:
        #     mlp1_dims = [int(x) for x in config.get('lstm_ga3c', 'mlp1_dims').split(', ')]
        #     self.model = ValueNetwork2(self.input_dim(), self.self_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        # else:
        #     self.model = ValueNetwork1(self.input_dim(), self.self_state_dim, mlp_dims, global_state_dim)

        self.multiagent_training = config.getboolean('lstm_ga3c', 'multiagent_training')
        logging.info('Policy: {}LSTM-GA3C {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        # print("inside lstm_ga3c predict:")
        # print(type(state)) #<class 'crowd_sim.envs.utils.state.JointState'>
        # print(state)

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        # return super().predict(state)


        # rewrite starts here
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
            # len(self.action_space)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                # VALUE UPDATE
                # print(self.model(rotated_batch_input)[1])
                next_state_value = self.model(rotated_batch_input)[1].data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        # print("inside model, predict")
        # print(self.phase)
        # print(state)
        # print(self.last_state)

        # print(max_action) # ActionXY(vx=-0.38268343236509034, vy=-0.9238795325112865)
        # print(type(max_action)) # <class 'crowd_sim.envs.utils.action.ActionXY'>

        return max_action

