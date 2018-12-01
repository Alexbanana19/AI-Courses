#!/usr/bin/env python

# modified from A3 gambler_exp.py

from rl_glue import *  # Required for RL-Glue
#RLGlue("windy_env", "n_step_sarsa_agent")
RLGlue("dyna_env", "dyna_agent")

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
    alphas = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

    results = np.zeros(len(alphas))

    for j in xrange(len(alphas)):
      print "Alpha: ", alphas[j]
      if j == 0:
        RL_agent_message("Alpha Start")
      else:
        RL_agent_message("Alpha")

      for i in xrange(num_runs):
        np.random.seed(i)
        RL_init()
        print "run ", i, "\n"

        sum_steps = 0
        for k in xrange(episodes):
          #print "Episodes ", k
          RL_start()
          is_terminal = RL_episode(max_steps)
          if is_terminal is True:
            steps = RL_num_steps()
            sum_steps += steps
            continue
        results[j] += 1.*sum_steps/episodes

    RL_cleanup()
    results = 1.*results/num_runs
    save_results(results, "results_dyna_0.2.pkl")
