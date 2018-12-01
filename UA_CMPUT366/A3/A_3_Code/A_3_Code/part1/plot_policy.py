import matplotlib.pyplot as plt
import pickle

with open("policy.pkl", "rb") as dataFile:
    p = pickle.loads(dataFile.read())


#print data
plt.step(range(1, len(p.tolist())-1),p.tolist()[1:100], label = 'opimal policy')
#plt.plot(range(len(eps01.tolist())),eps01.tolist(), label = 'realistic, Q1 = 0, eps = 0.1, alpha = 0.1')
plt.xlabel('Capital')
plt.ylabel('Final policy')
#plt.ylabel('Average rewards')
plt.title('Exercise 4.8')
plt.legend()
plt.show()
