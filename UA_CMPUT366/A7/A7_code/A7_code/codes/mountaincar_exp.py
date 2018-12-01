#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")

import numpy as np

def steps():
    num_episodes = 200
    num_runs = 50

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        np.random.seed(r)
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            steps[r,e] = RL_num_steps()
            #print steps[r,e]
    np.save('steps_1',steps)

def values():
    num_episodes = 1000
    num_runs = 1

    for r in range(num_runs):
        print "run number : ", r
        np.random.seed(r)
        RL_init()
        for e in range(num_episodes):
            RL_episode(0)

    values = RL_agent_message("Values")
    np.save('values_1.npy',values)
	
def bonus():
    num_episodes = 200
    num_runs = 50

    rewards = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        np.random.seed(r)
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
            rewards[r,e] = RL_return()
            #print rewards[r,e]
    np.save('rewards',rewards)

if __name__ == "__main__":
    steps()
    #values()
	#bonus()