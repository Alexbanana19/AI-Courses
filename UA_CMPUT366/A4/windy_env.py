#!/usr/bin/env python

# modified from A3 gambler_env.py
from utils import rand_norm, rand_in_range, rand_un
import numpy as np

current_state = None
wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

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

    if action == 'left':
    	x = x - 1    	
    	temp = (max(0, x), y)

    elif action == 'right':
    	x = x + 1
    	temp = (min(x, 9), y)

    elif action == 'down':
    	y = y - 1
    	temp = (x, max(0, y))

    elif action == 'up':
    	y = y + 1
    	temp = (x, min(y, 6))

    elif action == 'down-left':
    	x = x - 1
    	y = y - 1
    	temp = (max(0, x), max(0, y))

    elif action == 'down-right':
    	x = x + 1
    	y = y - 1
    	temp = (min(x, 9), max(0, y))

    elif action == 'up-left':
    	x = x - 1
    	y = y + 1
    	temp = (max(0, x), min(y, 6))

    elif action == 'up-right':
    	x = x + 1
    	y = y + 1
    	temp = (min(x, 9), min(y, 6))

    else:
    	temp = (x, y)
    
    y = temp[1] + wind[temp[0]] # wind 
    current_state = np.asarray([(temp[0], min(y, 6))])    

    
    reward = -1.0
    is_terminal = False
    if current_state[0][0] == 7 and current_state[0][1] == 3: 
        is_terminal = True
        current_state = None
        reward = 0.0

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
