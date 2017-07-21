#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:08:10 2017

@author: alex
"""
import klepto
import pickle
import tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/minghan/keras/lib/python2.7/site-packages')



#==============================================================================
#  loading data
#==============================================================================
with open('pretrained_lqr.pkl', 'rb') as f:
    data = pickle.loads(f.read())
state_and_action = data['s']
next_state = data['y']
X_train = state_and_action
Y_train = next_state
print X_train.shape, Y_train.shape

with open('pretrained_lqr.pkl', 'rb') as f:
    data = klepto.load(f.read())   
cost_grads = data['cost_gradients'] 
cost_hesses = data['cost_hessians']
print len(cost_grads), len(cost_hesses)
#==============================================================================
# NN structure
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
    x = tf.placeholder(tf.float32, [1, 393], name = 'x_input')
    y = tf.placeholder(tf.float32, [1, 376], name = 'y_input')

with tf.device('/gpu:0'):
    l1 = add_layer(x, 376+17, 2048, 1, tf.nn.sigmoid)    
    l2 = add_layer(l1, 2048, 2048, 2, tf.nn.sigmoid)
    l3 = add_layer(l2, 2048, 2048, 3, tf.nn.sigmoid)
    prediction = add_layer(l3, 2048, 376, 4)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    grad = tf.gradients(prediction, x)
#==============================================================================
# iLQR
#==============================================================================
saver = tf.train.Saver()
Session_config = tf.ConfigProto(allow_soft_placement=True)
Session_config.gpu_options.allow_growth = True
with tf.Session(config=Session_config) as sess:
  # Restore variables from disk.
    saver.restore(sess, "/home/alex/model/global/global.ckpt")
    K = []
    k = []
    V = np.zeros((393, 393))
    v = []
    Q = np.zeros((393, 393))
    q = np.zeros((1, 393))
    state_value = []
    q_value = []
    for i in xrange(X_train.shape[0]):
        f_grad = sess.run(grad, feed_dict={x: X_train[-i-1][None, :]})
        print f_grad.shape
        pr
        c_grad = cost_grads[i]
        c_hess = cost_hesses[i]
        

        
        
    
    
    
    
    
    