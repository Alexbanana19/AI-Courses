#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

  Experiment runs 2000 runs, each 1000 steps, of an n-armed bandit problem
"""

from rl_glue import *  # Required for RL-Glue
RLGlue("bandits_env", "bandits_agent")

import numpy as np
import sys
import pickle

def save_results(data, dataSize, filename): # data: floating point, dataSize: integer, filename: string
    with open(filename,'wb') as f:
        f.truncate()
    pickle.dump(data, open(filename, "wb"))

if __name__ == "__main__":
    num_runs = 2000
    max_steps = 1000

    # array to store the results of each step
    optimal_action = np.zeros(max_steps)
    #rewards = np.zeros(max_steps)
    print "\nPrinting one dot for every run: {0} total Runs to complete".format(num_runs)
    for k in range(num_runs):
        RL_init()

        obs = RL_start()
        opt = obs[0]
        for i in range(max_steps):
            # RL_step returns (reward, state, action, is_terminal); we need only the
            # action in this problem
            action = RL_step()[2]
            #reward = RL_step()[0]
            '''
            check if action taken was optimal

            you need to get the optimal action; see the news/notices
            announcement on eClass for how to implement this
            '''
            # update your optimal action statistic here
            if action == opt:
                optimal_action[i] += 1
            #rewards[i] += reward

        RL_cleanup()
        if k % 100 == 0:
            print "Episodes %d, optimal action %d"%(k, opt)

        sys.stdout.flush()

    save_results(1. * optimal_action / num_runs, max_steps, "optimistic.pkl")#optimistic.pkl
    print "\nDone"
