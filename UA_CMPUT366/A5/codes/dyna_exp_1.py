#!/usr/bin/env python

# modified from A3 gambler_exp.py

from rl_glue import *  # Required for RL-Glue
RLGlue("dyna_env", "dyna_prioritized_sweeping")

import numpy as np
import pickle

def save_results(data, filename):
    with open(filename,'wb') as f:
        f.truncate()
    pickle.dump(data, open(filename, "wb"))

if __name__ == "__main__":
    max_steps = 10000
    num_runs = 10
    episodes = 50

    results = np.zeros(episodes)
    for i in xrange(num_runs):
      np.random.seed(i)
      RL_init()
      print "run ", i, "\n"
      
      for j in xrange(episodes):
        print "Episodes ", j
        RL_start()
        is_terminal = RL_episode(max_steps)
        if is_terminal is True:
          steps = RL_num_steps()
          results[j] += steps
          continue

    RL_cleanup()
    results = 1.*results/num_runs
    save_results(results, "results_dyna_ps_5.pkl")
