#!/usr/bin/env python
# modified from A3 gambler_agent.py

# n-step sarsa agent
from utils import rand_in_range, rand_un
import numpy as np
import pickle

EPSILON = 0.1
GAMMA = 0.9
ALPHA = 0.5
N = 4

def agent_init():
    global q_values, actions, gammas
    actions = ['up', 'down', 'left', 'right']#,'up-left', 'up-right', 'down-left', 'down-right']#, 'none']
    q_values = np.zeros((70, len(actions)))
    gammas = np.zeros(N)
    for i in range(N):
        gammas[i] = GAMMA ** i

def agent_start(state):
    global actions, old_info, reward_buffer

    x = state[0][0]
    y = state[0][1]
    hash_state = y*10 + x

    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
        action = np.argmax(q_values[hash_state, :])

    old_info = []
    old_info.append((hash_state, action))
    reward_buffer = []

    return actions[action]


def agent_step(reward, state): 
    global q_values, actions, old_info, reward_buffer, gammas
    x = state[0][0]
    y = state[0][1]
    hash_state = y * 10 + x

    #epsilon greedy
    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
        action = np.argmax(q_values[hash_state, :])
    
    #learning
    reward_buffer.append(reward)

    if len(old_info) >= N:
        old_state = old_info[0][0]
        old_action = old_info[0][1]

        q_values[old_state, old_action] += ALPHA * (np.sum(gammas * np.asarray(reward_buffer))+\
            (GAMMA**(N)) * q_values[hash_state, action] - q_values[old_state, old_action])

        old_info.pop(0)
        reward_buffer.pop(0)

    old_info.append((hash_state, action))

    return actions[action]

def agent_end(reward):
    global q_values, actions, old_info, reward_buffer, gammas

    reward_buffer.append(reward)
    while len(old_info) > 0:
        n = len(old_info)
        old_state = old_info[0][0]
        old_action = old_info[0][1]

        q_values[old_state, old_action] += ALPHA * (np.sum(gammas[:n] * np.asarray(reward_buffer)) - q_values[old_state, old_action])

        old_info.pop(0)
        reward_buffer.pop(0)

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    return "I don't know what to return!!"

