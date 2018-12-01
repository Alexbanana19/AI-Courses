#!/usr/bin/env python
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("results_tabular.pkl", "rb") as dataFile:
    d0 = pickle.loads(dataFile.read())
with open("results_aggregation.pkl", "rb") as dataFile:
    d1 = pickle.loads(dataFile.read())
with open("results_tile_coding.pkl", "rb") as dataFile:
    d2 = pickle.loads(dataFile.read())
with open("results_fourier_10_0.001.pkl", "rb") as dataFile:
    d3 = pickle.loads(dataFile.read())
plt.plot(d0.tolist(), label = 'tabular features, alpha = 0.5')
plt.plot(d1.tolist(), label = 'state aggregation group 100, alpha = 0.1')
plt.plot(d2.tolist(), label = 'tile coding with 50 5x5 tilings, alpha = 0.01/50')
plt.plot(d3.tolist(), label = '10 order Fourier basis, alpha = 0.001')
plt.title('Semi-Gradient TD(0) with Different Basis Average over 10 Runs')

"""
with open("results_fourier_5_0.0001.pkl", "rb") as dataFile:
    d0 = pickle.loads(dataFile.read())
with open("results_fourier_10_0.0001.pkl", "rb") as dataFile:
    d1 = pickle.loads(dataFile.read())
with open("results_fourier_10_0.001.pkl", "rb") as dataFile:
    d2 = pickle.loads(dataFile.read())
with open("results_tile_coding.pkl", "rb") as dataFile:
    d3 = pickle.loads(dataFile.read())
plt.plot(d0.tolist(), label = '5 order Fourier basis, alpha = 0.0001')
plt.plot(d1.tolist(), label = '10 order Fourier basis, alpha = 0.0001')
plt.plot(d2.tolist(), label = '10 order Fourier basis, alpha = 0.001')
plt.plot(d3.tolist(), label = 'Tile Coding with 50 5x5 tilings, alpha = 0.01/50')
plt.title('Semi-Gradient TD(0) with Fourier Basis vs Tile Coding Average over 10 Runs')
"""

plt.xlabel('Episodes')
plt.ylabel('RMSE')

"""
with open("values_tabular.pkl", "rb") as dataFile:
    d0 = pickle.loads(dataFile.read())
with open("values_aggregation.pkl", "rb") as dataFile:
    d1 = pickle.loads(dataFile.read())
with open("values_tile_coding.pkl", "rb") as dataFile:
    d2 = pickle.loads(dataFile.read())
plt.plot(d0.tolist(), label = 'tabular features, alpha = 0.5')
plt.plot(d1.tolist(), label = 'state aggregation group 100, alpha = 0.1')
plt.plot(d2.tolist(), label = 'tile coding with 50 5x5 tilings, alpha = 0.01/50', color = 'red')
plt.title('State Values of Semi-Gradient TD(0) with Different Basis Average over 10 Runs')
plt.xlabel('State Value')
plt.ylabel('State')"""

plt.legend()
plt.show()
