#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:34:18 2017

@author: alex
"""

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
import load_policy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
args = parser.parse_args()

print('loading and building expert policy')
policy_fn = load_policy.load_policy(args.expert_policy_file)
print('loaded and built')


#==============================================================================
#  loading data
#==============================================================================

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
    x = tf.placeholder(tf.float32, [None, 376], name = 'x_input')
    y = tf.placeholder(tf.float32, [None, 17], name = 'y_input')


with tf.device('/gpu:0'):
    l1 = add_layer(x, 376, 2048, 1, tf.nn.sigmoid)
    l2 = add_layer(l1, 2048, 2048, 2, tf.nn.sigmoid)
    l3 = add_layer(l2, 2048, 2048, 2, tf.nn.sigmoid)
    prediction = add_layer(l3, 2048, 17, 3)
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
    with tf.variable_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

#==============================================================================
# Immitation Learning      
#============================================================================== 
env = gym.make(args.envname)
max_steps = args.max_timesteps or env.spec.timestep_limit

s = []
cost = []
y = []
saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
  # Restore variables from disk.
    saver.restore(sess, "/home/alex/model/imitation.ckpt")
    print("Model restored.")
    
    for i in xrange(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(prediction, feed_dict={x:obs[None,:]})
            s.append(obs.tolist()+action.tolist()[0])
            obs, r, done, _ = env.step(action)
            cost.append(-1.*r)
            y.append(obs)
            totalr += r
            steps += 1
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

    cost_data = {'s': np.array(s),
                       'cost': np.array(cost)}
    lqr_data = {'s': np.array(s),
                       'y': np.array(y)}
    with open('cost.pkl','wb') as f:
        f.truncate()
    pickle.dump(cost_data, open("cost.pkl", "wb"))
    
    with open('pretrained_lqr.pkl','wb') as f:
        f.truncate()
    pickle.dump(lqr_data, open("pretrained_lqr.pkl", "wb"))
    
    
