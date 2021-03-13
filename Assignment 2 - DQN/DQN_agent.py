import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# define parameters

env = gym.make('MountainCar-v0')
gamma = 0.9
max_epsilon = 0.99
min_epsilon = 0.1
epsilon_decay = 0.9995
max_memory_len = 10000
capacity = 20
update_frequency = 10
n_trials = 1000

# print action space, observation space
env_low = env.observation_space.low
env_high = env.observation_space.high

# create neural network

def NN(space_dim, n_actions, out_feature=24):
    return nn.Sequential(
        nn.Linear(space_dim, out_feature), 
        nn.ReLU(),
        nn.Linear(out_feature, out_feature),
        nn.ReLU(), 
        nn.Linear(out_feature, n_actions))

# create DQN class
class DQN:
    
    def __init__(self, env):
        # define parameters
        self.env = env
        self.epsilon = max_epsilon
        self.space_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        # initialize replace memory capacity
        self.memory = deque(maxlen = max_memory_len)
        
        # define policy & target networks
        self.policy = NN(self.space_dim, self.n_actions)
        self.target = NN(self.space_dim, self.n_actions)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        # NN evaluation metrics
        self.metric = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters())
        
    def state_to_float(self, state):
        '''Convert a state to float type'''
        return torch.from_numpy(np.reshape(state, [1, self.env.observation_space.shape[0]])).float()
    
    def add_memory(self, state, action, reward, next_state, terminal):
        '''Add new experience to memory'''
        self.memory.append((state, action, reward, next_state, terminal))
        
    def choose_action(self, state):
        '''Choose an action based on epsilon-greedy strategy'''
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.target(state).detach().numpy()[0])
            
        return action
    
    def epsilon_update(self):
        '''Decrease epsilon iteratively'''
        if self.epsilon > min_epsilon:
            self.epsilon *= epsilon_decay
        else:
            self.epsilon = min_epsilon
            
        return self.epsilon
    
    def target_update(self, cur_ep):
        '''Clone policy network weights to target network after a few time steps'''
        if cur_ep % update_frequency == 0:
            self.target.load_state_dict(self.policy.state_dict())
        
    def replay_memory(self):
        # randomly select n experiences from memory to train policy network
        if len(self.memory) < capacity:
            return
        
        batch = random.sample(self.memory, capacity)
        
        # update q value
        for state, action, reward, next_state, terminal in batch:
            if terminal and next_state[0][0].item() >= self.env.goal_position:
                q = reward
            
            else:
                q = reward + gamma * self.target(next_state).max(axis = 1)[0]
            # calculate new q-value
            q_val = self.target(state)
            q_val[0][action] = q
        
            # compute loss between output Q-value and target Q-value
            loss = self.metric(self.policy(state), q_val)

            # gradient descent 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()