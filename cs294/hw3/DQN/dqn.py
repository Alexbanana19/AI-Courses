#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:07:12 2017

@author: alex
"""
import sys
sys.path.append('/home/minghan/keras/lib/python2.7/site-packages')
from collections import deque
import tensorflow as tf
import numpy as np
import random
import cv2

UPDATE_FREQ = 10000
EPOCHS = 50000000
STEP_SIZE = 4
BATCH_SIZE = 32
MEM_SIZE = 1000000
REPLAY_START_SIZE = 50000
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
ANNEAL_SIZE = 1000000
NO_OP_MAX = 30

class DQN:
    def __init__(self, sess):
    # init replay memory
        self.replay_buffer = deque()
		# init some parameters
        self.steps = 0
        self.epsilon = INITIAL_EPSILON
		# init Q network
        self.x,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,\
    self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.create_model('/gpu:0')

		# init Target Q Network
        self.xT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,\
    self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.create_model('/gpu:2')
        self.createTrainingMethod()
        #copy
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),
            self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),
                self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),
                    self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),
                        self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
		# saving and loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
            self.save = True
        else:
            print "Could not find old network weights"   
            self.save = False
    def create_model(self, device):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 1.0)
            return tf.Variable(initial)
        
        def bias_variable(shape):
            initial = tf.constant(0.1, shape = shape)
            return tf.Variable(initial)
        
        def conv2d(x, W, strides):
            return tf.nn.conv2d(x, W, strides = [1,strides,strides,1], padding = 'VALID')
        
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        
        with tf.device(device):
            #No Max-pooling 
            x = tf.placeholder('float32', [None, 84, 84, 4])
            #keep_prob = tf.placeholder('float32')
            #convolutional layer 1
            W_conv1 = weight_variable([8,8,4,32])#patch 8x8, 32 filters
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 4) + b_conv1)#stride 4
            #convolutional layer 2
            W_conv2 = weight_variable([4,4,32,64])#patch 4x4, 64 filters
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)#stride 2
            #convolutional layer 2
            W_conv3 = weight_variable([3,3,64,64])#patch 3x3, 64 filters
            b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)#stride 1
            #fully connected layer 
            h_flat = tf.reshape(h_conv3, [-1, 7*7*64])
            W_fc1 = weight_variable([7*7*64, 512])
            b_fc1 = bias_variable([512])
            h_fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
            #output layer 
            W_fc2 = weight_variable([512,6])  
            b_fc2 = bias_variable([6])   
            prediction = tf.matmul(h_fc1, W_fc2) + b_fc2
            return x,prediction,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2   
        
    def save_model(self, sess):
        self.saver.save(sess, 'saved_networks/' + 'network' + '-dqn', global_step = self.steps)
        print "Model Saved" 
        
    def createTrainingMethod(self):
		self.a = tf.placeholder("float",[None,6])
		self.y = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.a), reduction_indices = 1)
		self.loss = tf.reduce_mean(tf.square(self.y - Q_Action))
		self.train_step = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)    
          
    def eps_greedy(self, action, num_no_op):
        rand = np.random.random()
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/ANNEAL_SIZE
        if self.epsilon < 0.1: self.epsilon = 0.1
        if rand < self.epsilon:
            if num_no_op <= NO_OP_MAX:
                return random.choice(range(6))
            else:
                return random.choice(range(2,6))
        else:
            if num_no_op <= NO_OP_MAX:
                return action
            else:
                return random.choice(range(2,6))
        
    def sampling(self):
        if len(self.replay_buffer) > BATCH_SIZE:
            if len(self.replay_buffer) > MEM_SIZE:
                self.replay_buffer.popleft()    
            return np.array([self.replay_buffer[random.choice(range(len(self.replay_buffer)))]\
                                                for i in xrange(BATCH_SIZE)])        
    
    def clipped_reward(self,reward):
        if reward > 0:
            return 1
        if reward < 0:
            return -1
        else :
            return 0
        
    def atari_reset(self, env):
        phi = []
        state = env.reset()
        action = 0
        for i in xrange(STEP_SIZE):
            self.steps += 1
            next_state, r, done, _ = env.step(action)
            phi.append(self.preprocessing(state))
            state = next_state
        
        return np.stack((phi[i]for i in xrange(STEP_SIZE)), axis = 2)
            
        
    def atari_step(self, phi, env, action):
        state = phi[-1]
        phi = []
        reward = 0
        for i in xrange(STEP_SIZE):
            self.steps += 1
            next_state, r, done, _ = env.step(action)
            state = next_state
            r = self.clipped_reward(r)
            phi.append(self.preprocessing(state))
            reward += r  
        return np.stack((phi[i]for i in xrange(STEP_SIZE)), axis = 2), reward, done
    
    def preprocessing(self, image):
        lum = cv2.cvtColor(cv2.resize(image, (84, 110)), cv2.COLOR_BGR2GRAY)
        lum = lum[26:110, :]#ger rid of the score board on the screen
        ret, thresh = cv2.threshold(lum, 1, 255, cv2.THRESH_BINARY)
        return thresh 
    