# Value Iteration
import numpy as np
import random
import pickle

GAMMA = 1
THETA = 0.001
Ph = 0.25
#SWEEPS = 32

def save_results(data, filename): 
    with open(filename,'wb') as f:
        f.truncate()
    pickle.dump(data, open(filename, "wb"))

class viAgent(object):
	def __init__(self, rewards, transitions):
		self.values = np.zeros(101)
		self.policy = np.ones(101)
		self.rewards = rewards
		self.transitions = transitions

	def update_values(self):
		delta = 99
		while delta > THETA:
			delta = 0
			for s in range(1, 100):
				v = self.values[s]
				max_value = -9999
				for a in xrange(1, min(s, 100-s) + 1):
					temp = (1-self.transitions) * (self.rewards[s-a] + GAMMA * self.values[s-a]) + \
							self.transitions * (self.rewards[s+a] + GAMMA * self.values[s+a])
					if max_value < temp:
						max_value = temp
				self.values[s] = max_value
				delta = max(delta, abs(self.values[s] - v))
			#print delta

	def update_policy(self):
		for s in range(1, 100):
			max_value = -9999
			max_action = 0
			for a in range(1, min(s, 100-s) + 1):
				temp = (1-self.transitions) * (self.rewards[s-a] + GAMMA * self.values[s-a]) + \
						self.transitions * (self.rewards[s+a] + GAMMA * self.values[s+a])
				if max_value < temp:
					max_value = temp
					max_action = a

			self.policy[s] = max_action


class environment(object):
	def __init__(self):
		self.states_num = 101

		self.terminal = np.zeros(101)
		self.terminal[0] = 1
		self.terminal[100] = 1

		self.rewards = np.zeros(101)
		self.rewards[100] = 1

		self.transitions = Ph

def main():
	env = environment()
	agent = viAgent(env.rewards, env.transitions) #pass the dynamics of the environment

	#for i in xrange(SWEEPS):
	agent.update_values()

	agent.update_policy()

	save_results(agent.values, 'values_delta.pkl')
	save_results(agent.policy, 'policy.pkl')



if __name__ == "__main__":
	main()
