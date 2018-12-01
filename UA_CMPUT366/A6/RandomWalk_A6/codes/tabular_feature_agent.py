#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random

ALPHA = 0.5
GAMMA = 1.0

def agent_init():
    global weights, actions, state_features
    # choose number of action
    actions = ['left', 'right']
    weights = np.zeros(1000) 
    state_features = np.zeros((1000,1000))
    state_features[np.arange(1000), np.arange(1000)] = 1  # one-hot encoding

def agent_start(state):
    global actions, old_state

    action = random.choice(actions)# 0.5 left and 0.5 right

    old_state = state[0]

    return action


def agent_step(reward, state):
	global actions, old_state, state_features, weights

	#learning
	v = np.sum(weights*(state_features[old_state-1]))
	nv = np.sum(weights*(state_features[state[0]-1]))
	weights += ALPHA*(reward+GAMMA*nv-v)*(state_features[old_state-1])

	old_state = state[0]
	action = random.choice(actions)
	return action

def agent_end(reward):
	global actions, old_state, state_features, weights

	#learning
	#print old_state
	v = np.sum(weights*(state_features[old_state-1]))

	weights += ALPHA*(reward-v)*(state_features[old_state-1])
	return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
	global state_features, weights
	if in_message == "Values":
		values = np.sum(weights*state_features,axis=1)
		return values
	else:
		return "I don't know what to return!!"
