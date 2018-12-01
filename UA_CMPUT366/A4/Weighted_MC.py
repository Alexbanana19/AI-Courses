#!/usr/bin/env python
# modified from A3 gambler_agent.py


from utils import rand_in_range, rand_un
import numpy as np
import pickle

EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.9

def agent_init():
    global q_values, actions, N
    actions = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right']#, 'none']
    q_values = np.zeros((70, len(actions)))
    N = np.zeros((70, len(actions)))

def agent_start(state):
    global actions, old_state, old_action, trajectory
    trajectory = []

    action = rand_in_range(len(actions))

    x = state[0][0]
    y = state[0][1]
    hash_state = y*10 + x # there are 70 states in total

    #print "start"
    #print state, action
    trajectory.append((hash_state, action))
    N[hash_state, action] += 1
    return actions[action]


def agent_step(reward, state):
    global q_values, actions, trajectory
    x = state[0][0]
    y = state[0][1]
    hash_state = y*10 + x

    #epsilon greedy
    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
        action = np.argmax(q_values[hash_state, :])
 
    #print state, action

    #learning
    n = len(trajectory)
    for i, (s, a) in enumerate(trajectory):
        #Wn = math.exp(-0.5*(g - (r+self.values[ns]))**2)
        q_values[s, a] += 1./N[s,a]*(GAMMA**(n-1-i))*(reward + GAMMA*q_values[hash_state, action] - q_values[s, a])

    old_state = hash_state
    old_action = action

    #if (hash_state, action) not in trajectory:
    trajectory.append((hash_state, action))
    N[hash_state, action] += 1
    return actions[action]

def agent_end(reward):
    global trajectory
    n = len(trajectory)
    for i, (s, a) in enumerate(trajectory):
        q_values[s, a] += 1./N[s,a]*(GAMMA**(n-1-i))*(reward - q_values[s, a])

    trajectory = []
    
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    return "I don't know what to return!!"

