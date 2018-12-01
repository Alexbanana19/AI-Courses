#!/usr/bin/env python
import matplotlib.pyplot as plt
import pickle

with open("results_dyna_0.pkl", "rb") as dataFile:
    d0 = pickle.loads(dataFile.read())
with open("results_dyna_5.pkl", "rb") as dataFile:
    d5 = pickle.loads(dataFile.read())
with open("results_dyna_50.pkl", "rb") as dataFile:
    d50 = pickle.loads(dataFile.read())
with open("results_dyna_0.2.pkl", "rb") as dataFile:
    d = pickle.loads(dataFile.read())
with open("results_dyna_ps_0.pkl", "rb") as dataFile:
    d_ps_0 = pickle.loads(dataFile.read())
with open("results_dyna_ps_5.pkl", "rb") as dataFile:
    d_ps_5 = pickle.loads(dataFile.read())
with open("results_dyna_ps_50.pkl", "rb") as dataFile:
    d_ps_50 = pickle.loads(dataFile.read())
#print data
#plt.plot(d0.tolist(), label = 'Dyna-Q 0 planning step')
#plt.plot(d5.tolist()[1:], label = 'Dyna-Q 5 planning step')
#plt.plot(d50.tolist(), label = 'Dyna-Q 50 planning step')
plt.plot([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0], d.tolist(), label = 'Dyna-Q 5 planning step Epsilon 0.2 average over 10 runs')
#plt.plot(d_ps_0.tolist(), label = "Dyna-Q 0 planning step with prioritized sweeping")
#plt.plot(d_ps_5.tolist()[1:], label = "Dyna-Q 5 planning step with prioritized sweeping")
#plt.plot(d_ps_50.tolist(), label = "Dyna-Q 50 planning step with prioritized sweeping")
#plt.plot([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0], d_ps.tolist(), label = 'Dyna-Q PS 5 planning step average over 10 runs')
#plt.yscale("log")
#plt.xlabel('Episodes')
#plt.ylabel('Steps')
plt.xlabel('Alpha')
plt.ylabel('Averaged Steps')

#plt.ylim(0, 200)
plt.title('Dyna-Q 5 averge over 10 runs')
#plt.title('Dyna-Q 5 vs Dyna-Q 5 with prioritized sweeping averge over 10 runs')
plt.legend()
plt.show()
