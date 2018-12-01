import matplotlib.pyplot as plt
import pickle

with open("values_delta.pkl", "rb") as dataFile:
    v = pickle.loads(dataFile.read())


#print data
plt.plot(range(1, len(v.tolist())-1),v.tolist()[1:100], label = 'value_iteration')
#plt.plot(range(len(eps01.tolist())),eps01.tolist(), label = 'realistic, Q1 = 0, eps = 0.1, alpha = 0.1')
plt.xlabel('Capital')
plt.ylabel('Value estimates')
#plt.ylabel('Average rewards')
plt.title('Ph = 0.25')
plt.legend()
plt.show()
