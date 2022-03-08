from collections import namedtuple
from random import sample, choice, randrange

from settings import *
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple


class ReplayMemory(object):  # Define a replay memory

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DNN(nn.Module):  # Double DQN
    def __init__(self, n_state, n_action):  # Define the layers of the fully-connected hidden network
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(n_state, 64)
        self.input_layer.weight.data.normal_(0, 0.1)
        self.middle_layer = nn.Linear(64, 32)
        self.middle_layer.weight.data.normal_(0, 0.1)
        self.middle_layer_2 = nn.Linear(32, 32)
        self.middle_layer_2.weight.data.normal_(0, 0.1)

        self.adv_layer = nn.Linear(32, n_action)
        self.adv_layer.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.middle_layer(x))
        x = F.relu(self.middle_layer_2(x))

        out = self.adv_layer(x)
        return out


class DQN:
    def __init__(self, n_action, n_state, learning_rate):

        self.n_action = n_action
        self.n_state = n_state

        self.memory = ReplayMemory(capacity=100)
        self.memory_counter = 0

        self.model_policy = DNN(self.n_state, self.n_action)
        self.model_target = DNN(self.n_state, self.n_action)
        self.model_target.load_state_dict(self.model_policy.state_dict())
        self.model_target.eval()

        self.optimizer = optim.Adam(self.model_policy.parameters(), lr=learning_rate)

    def store_transition(self, s, a, r, s_):
        state = torch.FloatTensor([s])
        action = torch.LongTensor([a])
        reward = torch.FloatTensor([r])
        next_state = torch.FloatTensor([s_])
        self.memory.Push(state, action, next_state, reward)

    def choose_action(self, state):
        state = torch.FloatTensor(state)
        if np.random.randn() <= EPISILO:  # greedy policy
            with torch.no_grad():
                q_value = self.model_policy(state)
                action = q_value.max(0)[1].view(1, 1).item()
        else:  # random policy
            action = torch.tensor([randrange(self.n_action)], dtype=torch.long).item()

        return action

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.Sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.model_policy(state_batch).gather(1, action_batch)

        next_action_batch = torch.unsqueeze(self.model_target(next_state_batch).max(1)[1], 1)
        next_state_values = self.model_target(next_state_batch).gather(1, next_action_batch)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.model_target.load_state_dict(self.model_policy.state_dict())
