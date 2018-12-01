import matplotlib.pyplot as plt
import pickle

with open("01EPS01.pkl", "rb") as dataFile:
    eps01 = pickle.loads(dataFile.read())
with open("optimistic.pkl", "rb") as dataFile:
    opt = pickle.loads(dataFile.read())


#print data
plt.plot(range(len(opt.tolist())),opt.tolist(), label = 'optimistic, Q1 = 5, eps = 0, alpha = 0.1')
plt.plot(range(len(eps01.tolist())),eps01.tolist(), label = 'realistic, Q1 = 0, eps = 0.1, alpha = 0.1')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
#plt.ylabel('Average rewards')
plt.title('Recreation of Figure 2.3')
plt.legend()
plt.show()
