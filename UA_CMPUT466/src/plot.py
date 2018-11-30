import matplotlib.pyplot as plt
import pickle
import numpy as np
d0 = []
d1 = []
d2 = []
for i in range(6):
    d0.append(np.load("run"+str(i)+"\\test\\teacher\\test_accuracy_lnin_none_run"+str(i)+".npy"))
for i in range(6):
    d1.append(np.load("run"+str(i)+"\\test\\teacher\\test_accuracy_dnin_none_run"+str(i)+".npy"))
for i in range(6):
    d2.append(np.load("run"+str(i)+"\\test\\teacher\\test_accuracy_mnin_none_run"+str(i)+".npy"))

d0 = np.asarray(d0).flatten()
d1 = np.asarray(d1).flatten()
d2 = np.asarray(d2).flatten()
print np.mean(d0), np.mean(d1), np.mean(d2)
print np.std(d0), np.std(d1), np.std(d2)
plt.scatter(np.arange(d0.shape[0]),d0, label="logits", color='r')
plt.scatter(np.arange(d1.shape[0]),d1, label="distillation", color='g')
plt.scatter(np.arange(d2.shape[0]),d2, label="multi-channel", color='b')

plt.legend()
plt.show()
