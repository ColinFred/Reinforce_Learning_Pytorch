from collections import namedtuple
from random import sample, choice, randrange, uniform

from settings import *
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayMemory_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity=1000, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, *args):
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        idxs = []
        segment = self.tree.total() / batch_size
        sample_datas = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = uniform(a, b)
            idx, p, data = self.tree.get(s)

            sample_datas.append(data)
            idxs.append(idx)
        return idxs, sample_datas

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries


class DNN(nn.Module):  # Dueling DQN
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

        self.value_layer = nn.Linear(32, 1)
        self.value_layer.weight.data.normal_(0, 0.1)

    def forward(self, state):  # Define the neural network forward function
        x = F.relu(self.input_layer(state))
        x = F.relu(self.middle_layer(x))
        x = F.relu(self.middle_layer_2(x))

        value = self.value_layer(x)
        adv = self.adv_layer(x)

        out = value + adv - adv.mean()
        return out


class PerDQN:
    def __init__(self, n_action, n_state, learning_rate):

        self.n_action = n_action
        self.n_state = n_state

        self.memory = ReplayMemory_Per(capacity=100)
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
        self.memory.push(state, action, next_state, reward)

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
        if self.memory.size() < BATCH_SIZE:
            return
        idxs, transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.model_policy(state_batch).gather(1, action_batch)

        next_action_batch = torch.unsqueeze(self.model_policy(next_state_batch).max(1)[1], 1)
        next_state_values = self.model_target(next_state_batch).gather(1, next_action_batch)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(1)

        td_errors = (state_action_values - expected_state_action_values).detach().squeeze().tolist()
        self.memory.update(idxs, td_errors)  # update td error
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.model_target.load_state_dict(self.model_policy.state_dict())
