#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    global policy, q_values, returns
    returns = {}
    policy = np.zeros(100)
    for s in xrange(1, 100):
        policy[s] = min(s, 100-s)
        for a in xrange(1, int(policy[s])+1):
            returns[(s, a)] = []
    q_values = np.zeros((100, 51))# 50 is the max_action
    #q_values = np.random.rand(101, 51)

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    global trajectory, policy
    trajectory = []
    s = state[0]
    action = rand_in_range(int(policy[s])) + 1
    trajectory.append([s, action])
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global Return, trajectory, policy

    #trajectory[-1].append(reward)

    action = int(policy[state[0]])
    print action

    flag = False
    for t in trajectory:
    	if t[0] == state[0] and t[1] == action:
    		flag = True
    		break
    
    if not flag:
    	trajectory.append([state[0], action])
    #print trajectory
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global trajectory, returns, policy, q_values

    
    #trajectory[-1].append(reward)

    #print trajectory
    for s, a in trajectory:
        returns[(s,a)].append(reward)# since the reward is 1 at the end and zero everywhere else
        q_values[s][a] = 1. * sum(returns[(s,a)])/len(returns[(s,a)])

    for s in xrange(1, 100):
        action = int(np.argmax(q_values[s, :]))
        if action != 0:
            policy[s] = min(action, min(s,100-s))

    trajectory = []
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

