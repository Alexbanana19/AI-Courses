#!/usr/bin/env python

# modified from A3 mc_agent.py
from utils import rand_in_range, rand_un
import numpy as np
import pickle
import random
import heapq

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.95
N = 5
THETA = 0.009
invalid_states = [20, 29, 38, 14, 34, 43, 52, 53]#for eliminating predecessors

def agent_init():
    global q_values, actions, pqueue, predecessors
    # choose number of action
    actions = ['up', 'down', 'left', 'right']
    q_values = np.zeros((54, len(actions)))  
    q_counter = np.zeros((54, len(actions)))
    pqueue = []
    predecessors = {}

def agent_start(state):
    global actions, old_state, old_action, q_values
    x = state[0][0]
    y = state[0][1]
    hash_state = y * 9 + x # there are 54 states in total

    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
        #breaking tie
        action = np.random.choice(np.where(q_values[hash_state, :] == q_values[hash_state, :].max())[0])

    old_state = hash_state
    old_action = action
    return actions[action]


def agent_step(reward, state):
    global q_values, actions, old_state, old_action, N, pqueue, predecessors
    x = state[0][0]
    y = state[0][1]
    hash_state = y * 9 + x

    priority = abs(reward + GAMMA*max(q_values[hash_state, :]) - q_values[old_state, old_action])
    if priority > THETA:
        heapq.heappush(pqueue, (-priority, (old_state, old_action, reward, hash_state)))
    #no learning

    #Prioritized Sweeping
    for i in xrange(N):
        if len(pqueue) == 0:
            break
        sample = heapq.heappop(pqueue)
        #print sample[0]
        s = sample[1][0]
        a = sample[1][1]
        r = sample[1][2]
        ns = sample[1][3]
        if ns is None:
            q_values[s,a]+=ALPHA*(r-q_values[s,a])
        else:
            q_values[s,a]+=ALPHA*(r+GAMMA*np.max(q_values[ns, :])-q_values[s,a])
        
        #compute predecessors
        if s not in predecessors: 
            x = s%9
            y = s/9
            predecessors[s] = []

            #checking boundary condition
            if x+1 <= 8:
                hash_ps = y * 9 + x + 1
                predecessors[s].append((hash_ps, 2, 0))#state, action, reward

            if x-1 >=0:
                hash_ps = y * 9 + x - 1
                predecessors[s].append((hash_ps, 3, 0))

            if y+1 <= 5:
                hash_ps = (y+1) * 9 + x
                predecessors[s].append((hash_ps, 1, 0))

            if y-1 >= 0:
                hash_ps = (y-1) * 9 + x
                predecessors[s].append((hash_ps, 0, 0))

        #sweep predecessors
        for pred in predecessors[s]:
            if pred[0] in invalid_states:# check invalid states
                continue
            os = pred[0]
            oa = pred[1]
            r = pred[2]
            priority = abs(r+GAMMA*np.max(q_values[s, :])-q_values[os,oa])
            if priority > THETA:
                heapq.heappush(pqueue, (-priority, (os, oa, r, s)))

    #epsilon greedy
    rand = rand_un()
    if rand <= EPSILON:
        action = rand_in_range(len(actions))
    else:
        action = np.random.choice(np.where(q_values[hash_state, :] == q_values[hash_state, :].max())[0])

    old_state = hash_state
    old_action = action

    return actions[action]

def agent_end(reward):
    global N, old_state, old_action, q_values, pqueue, actions, predecessors

    priority = abs(reward-q_values[old_state, old_action])
    if priority > THETA:
        heapq.heappush(pqueue, (-priority, (old_state, old_action, reward, None)))

    for i in xrange(N):
        if len(pqueue) == 0:
            break
        sample = heapq.heappop(pqueue)
        s = sample[1][0]
        a = sample[1][1]
        r = sample[1][2]
        ns = sample[1][3]
        if ns is None:
            q_values[s,a]+=ALPHA*(r-q_values[s,a])
        else:
            q_values[s,a]+=ALPHA*(r+GAMMA*np.max(q_values[ns, :])-q_values[s,a])
        
        #compute predecessors
        if s not in predecessors: 
            x = s%9
            y = s/9
            predecessors[s] = []
            if x+1 <= 8:
                hash_ps = y * 9 + x + 1
                predecessors[s].append((hash_ps, 2, 0))

            if x-1 >=0:
                hash_ps = y * 9 + x - 1
                predecessors[s].append((hash_ps, 3, 0))

            if y+1 <= 5:
                hash_ps = (y+1) * 9 + x
                predecessors[s].append((hash_ps, 1, 0))

            if y-1 >= 0:
                hash_ps = (y-1) * 9 + x
                predecessors[s].append((hash_ps, 0, 0))

        for pred in predecessors[s]:
            if pred[0] in invalid_states:
                continue
            os = pred[0]
            oa = pred[1]
            r = pred[2]
            priority = abs(r+GAMMA*np.max(q_values[s, :])-q_values[os,oa])
            if priority > THETA:
                heapq.heappush(pqueue, (-priority, (os, oa, r, s)))

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    
    return "I don't know what to return!!"
