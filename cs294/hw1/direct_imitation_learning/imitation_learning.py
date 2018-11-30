#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:45:13 2017

@author: alex
"""
import gym
import numpy as np
import tensorflow as tf
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
args = parser.parse_args()

Train_size = 40000
batch_size = 1000
#==============================================================================
#  loading data
#==============================================================================
with open('expert.pkl', 'rb') as f:
        data = pickle.loads(f.read())
obs = data['observations']
#print obs.shape
actions = data['actions'].reshape((obs.shape[0], 17))
X_train = obs[0:Train_size, :]
Y_train = actions[0:Train_size, :]

X_val = obs[Train_size:, :]
Y_val = actions[Train_size:, :]
#==============================================================================
# Training the Neural Network
#==============================================================================
def add_layer(inputs, in_size, out_size, n_layer, activation_fn = None):
    with tf.variable_scope('layer%d'%n_layer):
        W = tf.Variable(tf.random_normal([in_size, out_size]), name = 'weights')
        b = tf.Variable(tf.zeros([1, out_size]) + 0.01, name = 'biases')
        z = tf.add(tf.matmul(inputs, W), b, name = 'z')
        if activation_fn is None:
            outputs = z 
        else:
            outputs = activation_fn(z)
        return outputs
    

with tf.variable_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, obs.shape[1]], name = 'x_input')
    y = tf.placeholder(tf.float32, [None, actions.shape[1]], name = 'y_input')


with tf.device('/gpu:0'):
    l1 = add_layer(x, obs.shape[1], 200, 1, tf.nn.tanh)
    l2 = add_layer(l1, 200, 200, 2, tf.nn.sigmoid)
    l3 = add_layer(l2, 200, 200, 2, tf.nn.sigmoid)
    prediction = add_layer(l3, 200, 17, 3)
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
    with tf.variable_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)


#==============================================================================
# Immitation Learning      
#============================================================================== 
env = gym.make(args.envname)
max_steps = args.max_timesteps or env.spec.timestep_limit

returns = []
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
  # Restore variables from disk.
    saver.restore(sess, "/home/alex/model/imitation.ckpt")
    print("Model restored.")
    print sess.run([loss], feed_dict={x: X_val, y: Y_val})
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run([prediction], feed_dict={x:obs[None,:]})[0] 
            #print pred,type(pred), pred.shape
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
    #==============================================================================
            if args.render:
               env.render()
    #==============================================================================
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))