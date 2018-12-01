#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random
from tiles3 import tiles, IHT

MAX_SIZE = 2**12
TILE_SIZE = 0.2*1000
NUM_TILES = 1000/TILE_SIZE
NUM_TILINGS = 50
ALPHA = 0.01/NUM_TILINGS
GAMMA = 1.0

def agent_init():
    global actions, iht, weights
    # choose number of action
    actions = ['left', 'right']
    iht = IHT(MAX_SIZE)
    weights = np.zeros(MAX_SIZE)

def agent_start(state):
    global actions, old_state

    action = random.choice(actions)

    old_state = state[0]

    return action


def agent_step(reward, state):
    global actions, old_state, iht, weights

    #tile-coding
    scaled_s = 1.*NUM_TILES*(old_state-1)/999
    scaled_ns = 1.*NUM_TILES*(state[0]-1)/999
    hash_s = np.asarray(tiles(iht, NUM_TILINGS, [scaled_s]))
    hash_ns = np.asarray(tiles(iht, NUM_TILINGS, [scaled_ns]))

    v = np.sum(weights[hash_s])
    nv = np.sum(weights[hash_ns])

    #learning
    s_features = np.zeros_like(weights)
    s_features[hash_s] = 1.
    weights += ALPHA*(reward+GAMMA*nv-v)*s_features

    old_state = state[0]
    action = random.choice(actions)
    return action

def agent_end(reward):
    global actions, old_state, iht, weights

    #tile-coding
    scaled_s = 1.*NUM_TILES*(old_state-1)/999 # scaled the state to [0,10)
    hash_s = np.asarray(tiles(iht, NUM_TILINGS, [scaled_s]))

    #learning
    s_features = np.zeros_like(weights)
    s_features[hash_s] = 1
    v = np.sum(weights[hash_s])

    weights += ALPHA*(reward-v)*s_features

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
        estimates = np.zeros(1000)
        for s in range(1,1001):
            scaled_s = 1.*NUM_TILES/999*(s-1) # scaled the state to [0,10)
            hash_s = np.asarray(tiles(iht, NUM_TILINGS, [scaled_s]))
            v = np.sum(weights[hash_s])

            estimates[s-1]=v
        return estimates
    else:
        return "I don't know what to return!!"
