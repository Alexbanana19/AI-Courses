#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random
from math import cos
ALPHA = 0.001
GAMMA = 1.0
ORDER = 10
PI = 3.14

def agent_init():
    global weights, actions
    # choose number of action
    actions = ['left', 'right']
    weights = np.zeros(ORDER+1) 

def agent_start(state):
    global actions, old_state

    action = random.choice(actions)# 0.5 left and 0.5 right

    old_state = state[0]

    return action


def agent_step(reward, state):
	global actions, old_state, weights

	#constructing fourier features
	s_features = np.asarray([ cos(i*PI*old_state) for i in range(ORDER+1)])
	ns_features = np.asarray([cos(i*PI*state[0]) for i in range(ORDER+1)])
	#learning

	v = np.sum(weights*s_features)
	nv = np.sum(weights*ns_features)
	weights += ALPHA*(reward+GAMMA*nv-v)*s_features
	#print weights
	old_state = state[0]
	action = random.choice(actions)
	return action

def agent_end(reward):
	global actions, old_state, weights

	#constructing fourier features
	s_features = np.asarray([ cos(i*PI*old_state) for i in range(ORDER+1)])
	#print s_features
	#learning
	v = np.sum(weights*s_features)
	#print ALPHA*(reward-v)*s_features
	weights += ALPHA*(reward-v)*s_features
	#print weights
	return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
	global weights
	if in_message == "Values":
		estimates = np.zeros(1000)
		for i in range(1,1001):
			s_features = np.asarray([cos(j*PI*i) for j in range(ORDER+1)])
			estimates[i-1] = np.sum(weights*s_features)
		return estimates
	else:
		return "I don't know what to return!!"
