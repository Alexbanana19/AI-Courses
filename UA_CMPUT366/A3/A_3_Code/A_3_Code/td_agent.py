#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

ALPHA = .4
GAMMA = 1

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    global policy, q_values, returns
    q_values = np.zeros((100, 51))# 50 is the max_action
    #q_values = np.random.rand(101, 51)

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    global old_state, old_action
    s = state[0]
    action = rand_in_range(min(s, 100-s))+1
    #action = min(s, 100-s)
    old_action = action 
    old_state = s
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global q_values, old_state, old_action

    #trajectory[-1].append(reward)
    s = state[0]
    action = np.argmax(q_values[s,:])

    if action == 0 or action > min(s, 100-s):
        action = min(s, 100-s)
    
    #print action
    
    q_values[old_state][old_action] += ALPHA*(reward + GAMMA*q_values[s][action]-q_values[old_state][old_action])
    #print trajectory
    old_state = s
    old_action = action
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global q_values, old_state, old_action

    
    #trajectory[-1].append(reward)

    #print trajectory
    q_values[old_state][old_action] += ALPHA*(reward-q_values[old_state][old_action])

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global q_values
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(q_values, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"

