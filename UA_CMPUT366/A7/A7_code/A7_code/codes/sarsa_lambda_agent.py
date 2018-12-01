#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random
from tiles3 import tiles, IHT
"""
# original parameter setting
MAX_SIZE = 2**12
NUM_TILES = 8
NUM_TILINGS = 8
ALPHA = 0.1/NUM_TILINGS
EPSILON = 0.0
LAMBDA = 0.9
GAMMA = 1.0
"""
# Bonus question parameter setting
MAX_SIZE = 2**12
NUM_TILES = 8
NUM_TILINGS = 32
ALPHA = 1./NUM_TILINGS
EPSILON = 0.0
LAMBDA = 0.9
GAMMA = 1.0

POSITION = [-1.2, 0.6]
VELOCITY = [-0.07, 0.07]

def agent_init():
    global actions, iht, weights
    # choose number of action
    actions = [0,1,2]
    iht = IHT(MAX_SIZE)
    # 3xn matrix
    weights = np.random.uniform(-0.001,0,(len(actions), MAX_SIZE))

def agent_start(state):
    global actions, old_state, old_action, z
    #intialize eligibility trace
    z = np.zeros_like(weights)
    #tile-coding
    scaled_ns1 = 1.*NUM_TILES*(state[0]-POSITION[0])/(POSITION[1]-POSITION[0])
    scaled_ns2 = 1.*NUM_TILES*(state[1]-VELOCITY[0])/(VELOCITY[1]-VELOCITY[0])
    hash_ns = np.asarray(tiles(iht, NUM_TILINGS, [scaled_ns1, scaled_ns2]))

    #epsilon-greedy
    rand = rand_un()
    if rand < EPSILON:
        n_action = random.choice(actions)
    else:
        n_action = np.argmax(np.sum(weights[:,hash_ns],axis=1))

    old_state = hash_ns
    old_action = n_action

    return n_action


def agent_step(reward, state):
    global actions, old_state, old_action, iht, weights, z

    #tile-coding
    scaled_ns1 = 1.*NUM_TILES*(state[0]-POSITION[0])/(POSITION[1]-POSITION[0])
    scaled_ns2 = 1.*NUM_TILES*(state[1]-VELOCITY[0])/(VELOCITY[1]-VELOCITY[0])

    hash_s = old_state
    hash_ns = np.asarray(tiles(iht, NUM_TILINGS, [scaled_ns1, scaled_ns2]))

    #epsilon-greedy
    rand = rand_un()
    if rand < EPSILON:
        n_action = random.choice(actions)
    else:
        n_action = np.argmax(np.sum(weights[:,hash_ns],axis=1))

    #learning and update traces
    q = np.sum(weights[old_action,hash_s])
    nq = np.sum(weights[n_action,hash_ns])

    z[old_action,hash_s] = 1.

    weights += ALPHA*(reward+GAMMA*nq-q)*z
    z *= GAMMA*LAMBDA

    old_state = hash_ns
    old_action = n_action
    return n_action

def agent_end(reward):
    global actions, old_state, old_action, iht, weights, z

    #tile-coding
    hash_s = old_state

    #learning
    q = np.sum(weights[old_action,hash_s])
    z[old_action,hash_s] = 1.

    weights += ALPHA*(reward-q)*z

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global  weights, iht
    if in_message == "Values":
        values = np.zeros((50,50))
        for i in range(50):
            for j in range(50):
                pos = -1.2+i*1.7/50
                vel = -0.07+j*0.14/50
                scaled_s1 = 1.*NUM_TILES*(pos-POSITION[0])/(POSITION[1]-POSITION[0])
                scaled_s2 = 1.*NUM_TILES*(vel-VELOCITY[0])/(VELOCITY[1]-VELOCITY[0])
                hash_s = np.asarray(tiles(iht, NUM_TILINGS, [scaled_s1, scaled_s2]))
                q = np.max(np.sum(weights[:,hash_s],axis=1))
                values[i,j] = q
        return values
    else:
        return "I don't know what to return!!"
