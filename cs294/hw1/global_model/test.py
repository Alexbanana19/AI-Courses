#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 08:02:18 2017

@author: alex
"""

import pickle
import tensorflow as tf
import numpy as np
import gym


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    #==============================================================================
    #  loading data
    #==============================================================================
    with open('expert_lqr.pkl', 'rb') as f:
            data = pickle.loads(f.read())
    X = data['observations']
    Y = data['actions'].reshape((X.shape[0], 17))
    print X.shape
    
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
        l1 = add_layer(x, 376+17, 512, 1, tf.nn.sigmoid)    
        l2 = add_layer(l1, 512, 512, 2, tf.nn.sigmoid)
        l3 = add_layer(l2, 512, 512, 3, tf.nn.sigmoid)
        prediction = add_layer(l3, 512, 376, 4)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
        with tf.variable_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    
    saver = tf.train.Saver()
    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True


    
    #==============================================================================
    # Immitation Learning      
    #============================================================================== 
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    
    returns = []
    saver = tf.train.Saver()
    Session_config = tf.ConfigProto(allow_soft_placement=True)
    Session_config.gpu_options.allow_growth = True
    with tf.Session(config=Session_config) as sess:
      # Restore variables from disk.
        saver.restore(sess, "/home/alex/model/global/imitation.ckpt")
        print("Model restored.")
        
        for i in xrange(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run([prediction], feed_dict={x:obs[None,:]})[0] 
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
    
            returns.append(totalr)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
