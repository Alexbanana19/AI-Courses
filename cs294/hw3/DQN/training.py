#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:07:12 2017

@author: alex
"""
from dqn import *
import gym
#==============================================================================
# Model Training
#==============================================================================
ENV_NAME = 'Breakout-v0'
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
def main():
    env = gym.make(ENV_NAME)
    actions = np.zeros((1,6))   
    epoch = 0
    l = 0
    with tf.Session(config=config) as sess:
        dqn = DQN(sess) 
        if not dqn.save:
            sess.run(tf.initialize_all_variables())  
        while True:
            epoch += 1
            num_no_op = 0
            state = dqn.atari_reset(env)
            reward = 0
            while True:
                if len(dqn.replay_buffer) <= REPLAY_START_SIZE:
                    action = random.choice(range(6))
                else:
                    action = np.argmax(sess.run(dqn.QValue, feed_dict={dqn.x: state[None,:]}), axis = 1)[0]
                action = dqn.eps_greedy(action, num_no_op)
                if action == 0 or action == 1:
                    num_no_op += 4
                next_state, r, done = dqn.atari_step(state, env, action)
                reward += r
                a = np.zeros((6))
                a[action] = 1
                dqn.replay_buffer.append((state, a, r, next_state, done))
                state = next_state
                #model learning
                if len(dqn.replay_buffer) >= REPLAY_START_SIZE:
                    minibatch = dqn.sampling()
                    #rearrange the batch
                    state_batch = np.array([data[0] for data in minibatch])
                    action_batch = np.array([data[1] for data in minibatch])
                    reward_batch = np.array([data[2] for data in minibatch])
                    nextState_batch = np.array([data[3] for data in minibatch])                   
                    y_batch = []
                    QValue_batch = sess.run(dqn.QValueT, feed_dict={dqn.xT:nextState_batch})
                    for i in range(0,BATCH_SIZE):
                        terminal = minibatch[i][4]
                        if terminal:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
                    y_batch = np.array(y_batch)
                    train, l = sess.run([dqn.train_step,dqn.loss],feed_dict={dqn.y : y_batch, dqn.a : action_batch, \
                                            dqn.x : state_batch})
                    
                    if dqn.steps % 10000 == 0:
                        dqn.save_model(sess)
                    if dqn.steps % UPDATE_FREQ == 0:
                        print "Copying Q-network"
                        sess.run(dqn.copyTargetQNetworkOperation) 
                if done:
                    break  
            print "Epoch %d, Frame %d, training loss = %f, reward = %f"%(epoch, dqn.steps, l, reward)
                
if __name__ == '__main__':
    main()

