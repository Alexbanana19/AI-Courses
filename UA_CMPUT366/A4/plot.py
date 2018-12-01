#!/usr/bin/env python
import matplotlib.pyplot as plt
import pickle

with open("results9_01.pkl", "rb") as dataFile:
    data901 = pickle.loads(dataFile.read())

with open("results9_02.pkl", "rb") as dataFile:
    data902 = pickle.loads(dataFile.read())

with open("results9_03.pkl", "rb") as dataFile:
    data903 = pickle.loads(dataFile.read())

with open("results9_04.pkl", "rb") as dataFile:
    data904 = pickle.loads(dataFile.read())

with open("results9_05.pkl", "rb") as dataFile:
    data905 = pickle.loads(dataFile.read())

with open("results9_06.pkl", "rb") as dataFile:
    data906 = pickle.loads(dataFile.read())

with open("results9_07.pkl", "rb") as dataFile:
    data907 = pickle.loads(dataFile.read())

with open("results9_08.pkl", "rb") as dataFile:
    data908 = pickle.loads(dataFile.read())

with open("results9_09.pkl", "rb") as dataFile:
    data909 = pickle.loads(dataFile.read())

with open("results8.pkl", "rb") as dataFile:
    data8 = pickle.loads(dataFile.read())

with open("results8_06.pkl", "rb") as dataFile:
    data806 = pickle.loads(dataFile.read())

with open("results8_07.pkl", "rb") as dataFile:
    data807 = pickle.loads(dataFile.read())

with open("results8_08.pkl", "rb") as dataFile:
    data808 = pickle.loads(dataFile.read())

with open("results8n2.pkl", "rb") as dataFile:
    data8n2 = pickle.loads(dataFile.read())

with open("results8n3.pkl", "rb") as dataFile:
    data8n3 = pickle.loads(dataFile.read())

with open("results8n4.pkl", "rb") as dataFile:
    data8n4 = pickle.loads(dataFile.read())

with open("results8n8.pkl", "rb") as dataFile:
    data8n8 = pickle.loads(dataFile.read())

with open("results4_03.pkl", "rb") as dataFile:
    data403 = pickle.loads(dataFile.read())

with open("results4_04.pkl", "rb") as dataFile:
    data404 = pickle.loads(dataFile.read())

with open("results4_05.pkl", "rb") as dataFile:
    data405 = pickle.loads(dataFile.read())

with open("results4_06.pkl", "rb") as dataFile:
    data406 = pickle.loads(dataFile.read())

with open("results4_07.pkl", "rb") as dataFile:
    data407 = pickle.loads(dataFile.read())

with open("results4_08.pkl", "rb") as dataFile:
    data408 = pickle.loads(dataFile.read())

with open("results4n2.pkl", "rb") as dataFile:
    data4n2 = pickle.loads(dataFile.read())

with open("results4n3.pkl", "rb") as dataFile:
    data4n3 = pickle.loads(dataFile.read())

with open("results4n4.pkl", "rb") as dataFile:
    data4n4 = pickle.loads(dataFile.read())

#print data
#plt.plot(data4.tolist(), label = '1 step 4-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data406.tolist(), label = '1 step 4-moves, gamma = 0.9, alpha = 0.6, epsilon = 0.1')
#plt.plot(data407.tolist(), label = '1 step 4-moves, gamma = 0.9, alpha = 0.7, epsilon = 0.1')
#plt.plot(data408.tolist(), label = '1 step 4-moves, gamma = 0.9, alpha = 0.8, epsilon = 0.1')
#plt.plot(data4n2.tolist(), label = '2 step 4-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data4n3.tolist(), label = '3 step 4-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data4n4.tolist(), label = '4 step 4-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
plt.plot(data803.tolist(), label = '1 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
plt.plot(data804.tolist(), label = '1 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
plt.plot(data805.tolist(), label = '1 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
plt.plot(data806.tolist(), label = '1 step king-moves, gamma = 0.9, alpha = 0.6, epsilon = 0.1')
plt.plot(data807.tolist(), label = '1 step king-moves, gamma = 0.9, alpha = 0.7, epsilon = 0.1')
plt.plot(data808.tolist(), label = '1 step king-moves, gamma = 0.9, alpha = 0.8, epsilon = 0.1')
#plt.plot(data8n2.tolist(), label = '2 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data8n3.tolist(), label = '3 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data8n4.tolist(), label = '4 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data8n8.tolist(), label = '8 step king-moves, gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data901.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.1, epsilon = 0.1')
#plt.plot(data902.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.2, epsilon = 0.1')
#plt.plot(data903.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.3, epsilon = 0.1')
#plt.plot(data904.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.4, epsilon = 0.1')
#plt.plot(data905.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.5, epsilon = 0.1')
#plt.plot(data906.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.6, epsilon = 0.1')
#plt.plot(data907.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.7, epsilon = 0.1')
#plt.plot(data908.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.8, epsilon = 0.1')
#plt.plot(data909.tolist(), label = '1 step 9-moves(king-moves + rest), gamma = 0.9, alpha = 0.9, epsilon = 0.1')
plt.xlabel('Steps')
plt.ylabel('Episodes')
plt.title('1-Step 9 moves SARSA averge over 50 runs with different step size')
plt.legend()
plt.show()
