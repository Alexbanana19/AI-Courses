#!/usr/bin/env python

# modified from A3 gambler_exp.py

from rl_glue import *  # Required for RL-Glue
#RLGlue("windy_env", "n_step_sarsa_agent")
RLGlue("windy_env", "sarsa_agent")

import numpy as np
import pickle

def save_results(data, filename):
    with open(filename,'wb') as f:
        f.truncate()
    pickle.dump(data, open(filename, "wb"))

if __name__ == "__main__":
    max_steps = 8000
    num_runs = 50
    results = np.zeros(max_steps)
    for i in range(num_runs):
      RL_init()

      episodes = 0
      temp = []
      print "run ", i, "\n"
      
      RL_start()
      for i in range(max_steps):
        result = RL_step()
        if result['isTerminal'] is True:
          RL_start()
          episodes += 1
        temp.append(episodes)
      results += np.asarray(temp)
    print results.shape
    RL_cleanup()
    results = 1.*results/num_runs
    save_results(results, "results4_04.pkl")
