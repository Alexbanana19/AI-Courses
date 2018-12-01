#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

def smoothing(result, batch):#smooth the curves in the plot by averaging local batch
    r = []
    for j in xrange(result.shape[0]):
      new_result = []
      for i in xrange(result.shape[1]):
          average = 1.*sum(result[j, i: min(result.shape[1], i+batch)])/min(batch, result.shape[1] - i)
          new_result.append(average)
      r.append(new_result)
    return np.asarray(r)


if __name__ == "__main__":
   V = np.load('ValueFunction.npy')
   plt.show()
   
   #R = smoothing(V, 10)

   print V.shape, type(V)
   #for i, episode_num in enumerate([100, 1000, 8000]):
   
   plt.plot(V[0, :], label='episode : ' + str(100), alpha=1, color = 'blue')
   #plt.plot(R[0, :], color = 'blue')

   plt.plot(V[1, :], label='episode : ' + str(1000), alpha=1, color = 'green')
   #plt.plot(R[1, :], color = 'green')

   plt.plot(V[2, :], label='episode : ' + str(8000), alpha=1, color = 'red')
   #plt.plot(R[2, :], color = 'red')



   plt.xlim([0,100])
   plt.xticks([1,25,50,75,99])
   plt.xlabel('Capital')
   plt.ylabel('Value estimates')
   plt.title('Ph = 0.55')
   plt.legend()
   plt.show()