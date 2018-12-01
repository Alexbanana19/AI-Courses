#!/usr/bin/env python

# modified from A3 gambler_env.py
from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import random

def env_init():
    global current_state
    state = 500
    current_state = np.asarray([state])

def env_start():
    """ returns numpy array """
    global current_state

    state = 500
    current_state = np.asarray([state])
    return current_state

def env_step(action):
    global current_state

    reward = 0.0
    is_terminal = False

    #go left
    if action == 'left':
        left_stop = max(0, current_state[0]-100)
        left_range = current_state[0]-left_stop
        left_prob = 1.*(100-(left_range-1))/100
        rand = rand_un()
        if rand <= left_prob:
            current_state[0] = left_stop
        else:
            current_state[0] = random.choice(range(left_stop+1, current_state[0]))
    #go right
    else:
        right_stop = min(1001, current_state[0]+100)
        right_range = right_stop-current_state[0]
        right_prob = 1.*(100-(right_range-1))/100
        rand = rand_un()
        if rand <= right_prob:
            current_state[0] = right_stop

        else:
            current_state[0] = random.choice(range(current_state[0]+1,right_stop))

    if current_state[0] == 0: 
        is_terminal = True
        current_state = None
        reward = -1.0

    elif current_state[0] == 1001:
        is_terminal = True
        current_state = None
        reward = 1.0

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
