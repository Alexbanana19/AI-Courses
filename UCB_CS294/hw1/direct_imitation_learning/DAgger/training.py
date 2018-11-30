#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:44:55 2017

@author: alex
"""
import pickle
import tensorflow as tf
import numpy as np
#==============================================================================
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#==============================================================================
import sys
sys.path.append('/home/minghan/keras/lib/python2.7/site-packages')
import random
import numpy as np


#==============================================================================
#  loading data
#==============================================================================
with open('expert.pkl', 'rb') as f:
        data = pickle.loads(f.read())
obs = data['observations']
#print obs.shape
actions = data['actions'].reshape((obs.shape[0], 17))
X_train = obs
Y_train = actions
Train_size = X_train.shape[0]
batch_size = 2000
#==============================================================================
# Training the Neural Network
#==============================================================================
def add_layer(inputs, in_size, out_size, n_layer, activation_fn = None):
    with tf.variable_scope('layer%d'%n_layer):
        W = tf.Variable(tf.random_normal([in_size, out_size]), name = 'weights')
        b = tf.Variable(tf.zeros([1, out_size]) + 0.01, name = 'biases')
        z = tf.add(tf.matmul(inputs, W), b, name = 'z')
        drop_z = tf.nn.dropout(z, keep_prob)
        if activation_fn is None:
            outputs = drop_z 
        else:
            outputs = activation_fn(drop_z)
        return outputs
    

with tf.variable_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, obs.shape[1]], name = 'x_input')
    y = tf.placeholder(tf.float32, [None, actions.shape[1]], name = 'y_input')
    keep_prob = tf.placeholder(tf.float32)

with tf.device('/gpu:0'):
    l1 = add_layer(x, obs.shape[1], 2048, 1, tf.nn.sigmoid)    
    l2 = add_layer(l1, 2048, 2048, 2, tf.nn.sigmoid)
    l3 = add_layer(l2, 2048, 2048, 3, tf.nn.sigmoid)
    prediction = add_layer(l3, 2048, 17, 4)
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices = [1]))
    with tf.variable_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4 #只使用40%
with tf.Session(config=config) as sess:
    
    #saver.restore(sess, "/home/minghan/hw1/model/imitation.ckpt")
    sess.run(tf.initialize_all_variables())
    for epoch in xrange(2000):
        s = np.arange(X_train.shape[0])
        np.random.shuffle(s)
        X_train = X_train[s]
        Y_train = Y_train[s]
        
        for i in xrange(Train_size/batch_size):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            train, l = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
    
        if epoch % 10 == 0:
            print "Epoch %d, training loss = %f"%(epoch, l)
        if epoch % 500 == 0:
            save_path = saver.save(sess, "/home/minghan/hw1/model/imitation.ckpt")

#==============================================================================
# def TwoLayerModel():
#     model = Sequential()
#     model.add(Dense(100, input_dim = 376, init = 'normal', activation = 'relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(100, init = 'normal', activation = 'relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(17, init = 'normal'))
#     model.compile(loss = 'mse', optimizer = 'adam')
#     return model
# 
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=TwoLayerModel, nb_epoch=50, batch_size=32, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, obs, actions, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#==============================================================================

