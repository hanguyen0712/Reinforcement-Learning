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
q_table_sarsa = np.zeros((bins, bins, env.action_space.n))

# store reward
rewards_sarsa = []

# training 

for ep in range(n_eps):
    
    current_reward = 0
    done = False

    state = env.reset()
    # discretize the state
    pos, vel = getState(state)
    # choose first action
    action = chooseAction(pos, vel, q_table_sarsa)

    while not done:
        
        # render for the last 10 episodes
        if ep >= (n_eps - 10): 
            env.render()
            
        # next state
        next_state, reward, done, info = env.step(action)

        # discretize the state
        next_pos, next_vel = getState(next_state)
        # next action
        next_action = chooseAction(next_pos, next_vel, q_table_sarsa)

        if done and next_state[0] >= env.goal_position:
            q_table_q[next_pos][next_vel][action] = reward
        
        else:
            # update Q value: Q(S, A) <-- Q(S, A) + alpha [R + gamma * Q(S', A') - Q(S, A)]
            q_table_sarsa[pos][vel][action] += \
            alpha * (reward + gamma * q_table_sarsa[next_pos][next_vel][next_action] - q_table_sarsa[pos][vel][action])
        
        # reassign state, action, reward
        state = next_state
        pos, vel = next_pos, next_vel
        action = next_action
        current_reward += reward
    
    # update epsilon
    if epsilon > 0:
        epsilon*= (n_eps - 1)/n_eps

    if ep % interval == 0:
        print('Game no.: ', ep, 'epsilon: ', epsilon, 'with reward: ', current_reward)
    rewards_sarsa.append(current_reward)

env.close()