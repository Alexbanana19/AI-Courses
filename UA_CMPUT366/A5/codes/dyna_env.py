#!/usr/bin/env python

# modified from A3 gambler_env.py
from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None
blocks = [(2,2), (2,3), (2,4), (5,1), (7,3), (7,4), (7,5)]
def env_init():
    global current_state
    state = (0, 3)
    current_state = np.asarray([state])

def env_start():
    """ returns numpy array """
    global current_state

    state = (0, 3)
    current_state = np.asarray([state])
    return current_state

def env_step(action):
    global current_state

    x = current_state[0][0]
    y = current_state[0][1]

    # transition and check the boundary conditions
    if action == 'left':
    	x = x - 1    	
    	temp = (max(0, x), y)

    elif action == 'right':
    	x = x + 1
    	temp = (min(x, 8), y)

    elif action == 'down':
    	y = y - 1
    	temp = (x, max(0, y))

    elif action == 'up':
    	y = y + 1
    	temp = (x, min(y, 5))

    else:
    	temp = (x, y)

    reward = 0.0
    is_terminal = False

    if temp[0] == 8 and temp[1] == 5: 
        is_terminal = True
        current_state = None
        reward = 1.0
        result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    elif temp in blocks:
        result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    else:
        current_state = np.asarray([(temp[0], temp[1])])
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
