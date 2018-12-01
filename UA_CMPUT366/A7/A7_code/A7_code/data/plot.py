import os
from matplotlib import pyplot as plt
import numpy as np

filename = 'steps.npy'
filename1 = 'steps_1.npy'
if os.path.exists(filename):
    data = np.load(filename)
    data1 = np.load(filename1)
    lmda = 0.90
    d = np.mean(data,axis=1)
    d1 = np.mean(data1,axis=1)
    print np.mean(d), np.mean(d1)
    print np.std(d), np.std(d1)
    plt.plot(np.arange(1,data.shape[1]+1),np.mean(data,axis=0),label='Original Parameter Setting Sarsa_lambda={}'.format(lmda))
    plt.plot(np.arange(1,data1.shape[1]+1),np.mean(data1,axis=0),label='Bonus Question Patameter Setting Sarsa_lambda={}'.format(0.9))
    plt.ylim([100,500])
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode \naveraged over {} runs'.format(data.shape[0]))
    plt.title("Averge Steps over 50 runs")	
    plt.legend()
    plt.show()
