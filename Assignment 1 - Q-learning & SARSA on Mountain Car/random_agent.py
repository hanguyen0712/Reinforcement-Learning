import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

# make gym environment
env = gym.make('MountainCar-v0')

# print action space, observation space

n_action = env.action_space.n
env_low = env.observation_space.low
env_high = env.observation_space.high
bins = 30 # number of states for discretization

# space discretization

def getState(state, env_low = env_low, env_high = env_high, bins = bins):
    """Returns the discretized position and velocity of an observation"""
    discretized_env = (env_high - env_low) / bins
    discretized_pos = int((state[0] - env_low[0]) / discretized_env[0])
    discretized_vel = int((state[1] - env_low[1]) / discretized_env[1])
    return discretized_pos, discretized_vel

n_eps = 100001 # number of episodes
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.2 # explore-exploit tradeoff factor
interval = 10000

# establish q-table
q_table_rand = np.zeros((bins, bins, env.action_space.n))

# store reward
rewards_rand = []

# training 

for ep in range(n_eps):
    
    state = env.reset()
    current_reward = 0
    done = False

    # discretize the state
    pos, vel = getState(state)

    while not done:
        
        # render for the last 10 episodes
        if ep >= (n_eps - 10): 
            env.render()

        # next action
        action = env.action_space.sample()

        # next state
        next_state, reward, done, info = env.step(action)
        # discretize next state
        next_pos, next_vel = getState(next_state)

        if done and next_state[0] >= env.goal_position:
            q_table_q[next_pos][next_vel][action] = reward
        
        else:
            # update Q value: Q(S, A) <-- Q(S, A) + alpha [R + gamma * Q(S', A') - Q(S, A)]
            q_table_q[pos][vel][action] += \
            alpha * (reward + gamma * np.max(q_table_q[next_pos][next_vel]) - q_table_q[pos][vel][action])
        
            
        # reassign state, action, reward
        state = next_state
        pos, vel = next_pos, next_vel
        current_reward += reward

    if ep % interval == 0:
        print('Game no.: ', ep, 'epsilon: ', epsilon, 'with reward: ', current_reward)
    rewards_rand.append(current_reward)

env.close()