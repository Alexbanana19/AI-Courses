#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 22:24:59 2017

@author: alex
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:46:07 2017

@author: alex
"""

import pickle
import tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/minghan/keras/lib/python2.7/site-packages')
import random
import numpy as np


#==============================================================================
#  loading data
#==============================================================================
with open('cost.pkl', 'rb') as f:
        data = pickle.loads(f.read())
state_and_action = data['s']
cost = data['cost']
X_train = state_and_action
Y_train = cost[:,None]
print X_train.shape, Y_train.shape
Train_size = X_train.shape[0]
batch_size = 2000
#==============================================================================
# learn cost function
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
    x = tf.placeholder(tf.float32, [None, 393], name = 'x_input')
    y = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

with tf.device('/gpu:0'):
    l1 = add_layer(x, 376+17, 256, 1, tf.nn.sigmoid)    
    l2 = add_layer(l1, 256, 256, 2, tf.nn.sigmoid)
    prediction = add_layer(l2, 256, 1, 4)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        
saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4 #只使用40%
with tf.Session(config=config) as sess:
    #saver.restore(sess, "/home/minghan/hw1/model/imitation.ckpt")
    sess.run(tf.initialize_all_variables())
    for epoch in xrange(2000):
        s = np.arange(X_train.shape[0])
        #np.random.shuffle(s)
        #X_train = X_train[s]
        #Y_train = Y_train[s]
        for i in xrange(Train_size/batch_size):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            train, l = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
    
        if epoch % 10 == 0:
            print "Epoch %d, training loss = %f"%(epoch, l)
        if epoch % 500 == 0:
            save_path = saver.save(sess, "/home/minghan/hw1/model/cost/cost.ckpt")