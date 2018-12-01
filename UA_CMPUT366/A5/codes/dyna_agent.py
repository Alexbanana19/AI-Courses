#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random

EPSILON = 0.2
ALPHA = 0.1 # part 1
GAMMA = 0.95
N = 5
alphas = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]# for part II

def agent_init():
    global q_values, actions, sample_buffer
    # choose number of action
    actions = ['up', 'down', 'left', 'right']
    q_values = np.zeros((54, len(actions)))  
    sample_buffer = [] 

def agent_start(state):
    global actions, old_state, old_action, q_values

    x = state[0][0]
    y = state[0][1]
    hash_state = y * 9 + x # there are 54 states in total

    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
    	#breaking tie
    	action = np.random.choice(np.where(q_values[hash_state, :] == q_values[hash_state, :].max())[0])

    old_state = hash_state
    old_action = action

    return actions[action]


def agent_step(reward, state):
	global q_values, actions, old_state, old_action, N, sample_buffer, count
	x = state[0][0]
	y = state[0][1]
	hash_state = y * 9 + x

	sample_buffer.append((old_state, old_action, reward, hash_state))

	#learning
	ALPHA = alphas[count]
	q_values[old_state, old_action] += ALPHA * (reward + GAMMA*max(q_values[hash_state, :]) - q_values[old_state, old_action])
	
	#planning
	for i in xrange(N):
		sample = random.choice(sample_buffer)
		s = sample[0]
		a = sample[1]
		r = sample[2]
		ns = sample[3]
		if ns is None:
			q_values[s,a]+=ALPHA*(r-q_values[s,a])
		else:
			q_values[s,a]+=ALPHA*(r+GAMMA*np.max(q_values[ns, :])-q_values[s,a])

	#epsilon greedy
	rand = rand_un()
	if rand <= EPSILON:
	    action = rand_in_range(len(actions))
	else:
	    #action = np.argmax(q_values[hash_state, :])
	    action = np.random.choice(np.where(q_values[hash_state, :] == q_values[hash_state, :].max())[0])

		

	old_state = hash_state
	old_action = action
	return actions[action]

def agent_end(reward):
    global N, old_state, old_action, q_values, count

    sample_buffer.append((old_state, old_action, reward, None))
    #learning
    ALPHA = alphas[count]
    q_values[old_state, old_action] += ALPHA * (reward - q_values[old_state, old_action])
    #planning
    for i in xrange(N):
    	sample = random.choice(sample_buffer)
    	s = sample[0]
    	a = sample[1]
    	r = sample[2]
    	ns = sample[3]
    	if ns is None:
    		q_values[s,a]+=ALPHA*(r-q_values[s,a])
    	else:
    		q_values[s,a]+=ALPHA*(r+GAMMA*np.max(q_values[ns, :])-q_values[s,a])
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
	global count
	# use for setting alpha
	if in_message == "Alpha Start":
		count = 0
	elif in_message == "Alpha":
		count += 1
	else:
		return "I don't know what to return!!"
