#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle

EPSILON = 0.1
ALPHA = 0.4
GAMMA = 0.9

def agent_init():
    global q_values, actions, N
    # choose number of action
    actions = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right']#, 'none']
    q_values = np.zeros((70, len(actions)))
    N = np.zeros((70, len(actions)))    


def agent_start(state):
    global actions, old_state, old_action, q_values

    x = state[0][0]
    y = state[0][1]
    hash_state = y * 10 + x # there are 70 states in total

    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
        action = np.argmax(q_values[hash_state, :])

    old_state = hash_state
    old_action = action

    return actions[action]


def agent_step(reward, state):
    global q_values, actions, old_state, old_action, N
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
    q_values[old_state, old_action] += ALPHA * (reward + GAMMA*q_values[hash_state, action] - q_values[old_state, old_action])

    old_state = hash_state
    old_action = action

    return actions[action]

def agent_end(reward):
    global N, old_state, old_action, q_values
    q_values[old_state, old_action] += ALPHA * (reward - q_values[old_state, old_action])

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    return "I don't know what to return!!"

