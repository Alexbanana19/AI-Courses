#!/usr/bin/env python

# modified from A3 gambler_exp.py

from rl_glue import *  # Required for RL-Glue

#RLGlue("rndmwalk_1000_env", "tile_coding_agent")
#RLGlue("rndmwalk_1000_env", "tabular_feature_agent")
RLGlue("rndmwalk_1000_env", "state_aggregation_agent")
#RLGlue("rndmwalk_1000_env", "fourier_agent")

from rndmwalk_policy_evaluation import compute_value_function
import numpy as np
import pickle

def save_results(data, filename):
    with open(filename,'wb') as f:
        f.truncate()
    pickle.dump(data, open(filename, "wb"))

if __name__ == "__main__":
    max_steps = 100000
    num_runs = 10
    episodes = 5000

    #with open("TrueValueFunction.npy", "rb") as dataFile:
    values = np.load("TrueValueFunction.npy")[1:1001]

    results = np.zeros(episodes)
    for i in xrange(num_runs):
      np.random.seed(i)
      RL_init()
      print "run ", i, "\n"
      
      for j in xrange(episodes):
        #print "Episodes ", j
        RL_start()
        is_terminal = RL_episode(max_steps)
        if is_terminal is True:
          estimates = RL_agent_message("Values")
          error = (1./1000*np.sum(np.square(values-estimates)))**0.5
          #print error
          results[j] += error
          continue

    RL_cleanup()
    results = 1.*results/num_runs
    save_results(results, "results_aggregation.pkl")
    save_results(estimates, "values_aggregation.pkl")
