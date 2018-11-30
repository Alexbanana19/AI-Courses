import numpy as np
import tensorflow as tf
from copy import deepcopy
import random
import os

batch_size = 128
temperature = 1.0

class SSAE(object):
    def __init__(self, x_dims, y_dims, sess, choice):
        self.sess = sess
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.choice = choice
        self.build_model()
        self.saver = tf.train.Saver(
            [v for v in tf.all_variables() if 'ssae' in v.name])

        if not os.path.exists("saved_model/ssae"):
            os.makedirs("saved_model/ssae")

        checkpoint = tf.train.get_checkpoint_state("saved_model/ssae/")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            print "Model Restored:", checkpoint.model_checkpoint_path
            self.save = True
        else:
            print "Could not find old network weights"
            self.save = False

    def build_model(self):
        def weight_variable(shape):
            W = tf.get_variable(
                'W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            return W

        def bias_variable(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def conv1d(x, W, s1, s2):
            return tf.nn.conv2d(x, W, strides=[1, s1, s2, 1], padding='VALID')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
        with tf.device('/gpu:3'):
            with tf.variable_scope('ssae'):
                with tf.variable_scope('inputs'):
                    self.X_input = tf.placeholder('float32', [None, 128])
                    self.y_input = tf.placeholder('float32', [None, 10])
                    self.keep_prob = tf.placeholder(tf.float32)
                    count = 0
                    self.mask = np.zeros((128,16))
                    for i in range(16):
                        for j in range(8):
                            self.mask[count+j][i] = 1.0
                        count += 8
                    self.mask = np.tile(self.mask, [1,10])

                with tf.variable_scope('conv_layer_1'):
                    self.W_conv1 = weight_variable([128, 160])*self.mask
                    self.b_conv1 = bias_variable([160])
                    self.sublogits = tf.reshape(tf.matmul(self.X_input, self.W_conv1) + self.b_conv1, [tf.shape(self.X_input)[0], 10, 16])
                    self.subpred = tf.nn.sigmoid(self.sublogits)
                    self.embedding = tf.contrib.layers.flatten(tf.reduce_mean(tf.reshape(self.subpred,[tf.shape(self.X_input)[0], 10, 4, 4]),axis=3))

                with tf.variable_scope('softmax'):
                    self.logits = tf.reduce_mean(self.subpred, axis=2)
                    self.prediction = tf.nn.softmax(self.logits)

                with tf.variable_scope('train'):
                    self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.logits))
                    self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def get_embedding(self, x):
        embedding = self.sess.run(self.subpred, feed_dict={self.X_input: x, self.keep_prob: 1.0})
        embedding = embedding.reshape((embedding.shape[0],160))
        return embedding

    def predict(self, x):
        pred = self.sess.run(self.prediction, feed_dict={self.X_input: x, self.keep_prob: 1.0})
        return pred

    def accuracy(self, pred, y):
        correct = 0
        for i in range(pred.shape[0]):
            if np.argmax(pred[i]) == np.argmax(y[i]):
                correct += 1

        accuracy = 1. * correct / pred.shape[0]

        return accuracy

    def test(self, X_test, y_test):
        pred = self.predict(X_test)
        accuracy = self.accuracy(pred, y_test)

        return accuracy

    def train(self, X_train, y_train):
        train, l, prediction = self.sess.run([self.train_step, self.loss, self.prediction],\
                                             feed_dict={self.X_input: X_train, self.y_input: y_train, self.keep_prob: 1.0})

        a = self.accuracy(prediction, y_train)
        return l, a

    def restore_model(self):
        self.saver.restore(self.sess, 'saved_model/ssae/ssae.ckpt')
        print "Model Restored"

    def save_model(self):
        self.saver.save(self.sess, 'saved_model/ssae/ssae.ckpt')
        print "Model Saved"

class AE(object):
    def __init__(self, x_dims, y_dims, sess, choice):
        self.sess = sess
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.choice = choice
        self.build_model()
        self.saver = tf.train.Saver(
            [v for v in tf.all_variables() if 'ae' in v.name])

        if not os.path.exists("saved_model/ae"):
            os.makedirs("saved_model/ae")

        checkpoint = tf.train.get_checkpoint_state("saved_model/ae/")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            print "Model Restored:", checkpoint.model_checkpoint_path
            self.save = True
        else:
            print "Could not find old network weights"
            self.save = False

    def build_model(self):
        def weight_variable(shape):
            W = tf.get_variable(
                'W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            return W

        def bias_variable(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def conv1d(x, W, s1, s2):
            return tf.nn.conv2d(x, W, strides=[1, s1, s2, 1], padding='VALID')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
        with tf.device('/gpu:3'):
            with tf.variable_scope('ae'):
                with tf.variable_scope('inputs'):
                    self.X_input = tf.placeholder('float32', [None, 256])

                with tf.variable_scope('encoder_1'):
                    self.W_ec1 = weight_variable([256, 128])
                    self.b_ec1 = bias_variable([128])
                    self.embedding = tf.nn.sigmoid(tf.matmul(self.X_input,self.W_ec1) + self.b_ec1)

                with tf.variable_scope('decoder_1'):
                    self.W_dc1 = weight_variable([128, 256])
                    self.b_dc1 = bias_variable([256])
                    self.recon = tf.matmul(self.embedding, self.W_dc1) + self.b_dc1

                with tf.variable_scope('train'):
                    self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.X_input-self.recon),axis=1),axis=0)
                    self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def get_embedding(self, x):
        labels = self.sess.run(self.embedding, feed_dict={self.X_input: x})
        return labels

    def train(self, X_train):
        train, l = self.sess.run([self.train_step, self.loss],feed_dict={self.X_input: X_train})

        a = 0
        return l, a

    def restore_model(self):
        self.saver.restore(self.sess, 'saved_model/ae/ae.ckpt')
        print "Model Restored"

    def save_model(self):
        self.saver.save(self.sess, 'saved_model/ae/ae.ckpt')
        print "Model Saved"


class SN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'sn' in v.name])

		if not os.path.exists("saved_model/sn"):
			os.makedirs("saved_model/sn")

		checkpoint = tf.train.get_checkpoint_state("saved_model/sn/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.variable_scope('sn'):
			with tf.variable_scope('inputs'):
				self.X_input = tf.placeholder('float32', [None,32,32,3])
				self.y_input = tf.placeholder('float32', [None, 10])
				self.keep_prob = tf.placeholder(tf.float32)

			with tf.variable_scope('conv_layer_1'):
				self.W_conv1=weight_variable([3,3,3,128])
				self.b_conv1=bias_variable([128])
				self.h_conv1=conv2d(self.X_input,self.W_conv1)+self.b_conv1
				self.h_pool1=max_pool_2x2(self.h_conv1)
				#self.h_drop1 = tf.nn.dropout(self.h_pool1, keep_prob=self.keep_prob)
				self.h_pool1_flat = tf.contrib.layers.flatten(self.h_pool1)

			with tf.variable_scope('linear_bottleneck'):
				self.W_fc0 = weight_variable([16*16*128,1200])
				self.b_fc0 = bias_variable([1200])
				self.h_fc0 = tf.matmul(self.h_pool1_flat, self.W_fc0) + self.b_fc0

			with tf.variable_scope('fc_layer_1'):
				self.W_fc1 = weight_variable([1200,1024*8])
				self.b_fc1 = bias_variable([1024*8])
				self.h_fc1 = tf.nn.relu(tf.matmul(self.h_fc0, self.W_fc1) + self.b_fc1)
				#self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)

			with tf.variable_scope('softmax'):
				self.W_fc2 = weight_variable([1024*8, 10])
				self.b_fc2 = bias_variable([10])
				self.logits = tf.matmul(self.h_fc1,self.W_fc2) + self.b_fc2
				self.prediction = tf.nn.softmax(self.logits)

			with tf.variable_scope('train'):
				self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_input, logits = self.logits))
				self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def predict(self, x):
		prediction = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return prediction

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train):
		train, l = self.sess.run([self.train_step, self.loss],\
	 		feed_dict = {self.X_input: X_train, self.y_input: y_train, self.keep_prob: 1.0})
		prediction = self.predict(X_train)
		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/sn/sn.ckpt')
        	print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/sn/sn.ckpt')
        	print "Model Saved"

class LSN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'lsn' in v.name])

		if not os.path.exists("saved_model/lsn"):
			os.makedirs("saved_model/lsn")

		checkpoint = tf.train.get_checkpoint_state("saved_model/lsn/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.variable_scope('lsn'):
			with tf.variable_scope('inputs'):
				self.X_input = tf.placeholder('float32', [None,32,32,3])
				self.y_input = tf.placeholder('float32', [None, 10])
				self.keep_prob = tf.placeholder(tf.float32)

			with tf.variable_scope('conv_layer_1'):
				self.W_conv1=weight_variable([3,3,3,128])
				self.b_conv1=bias_variable([128])
				self.h_conv1=conv2d(self.X_input,self.W_conv1)+self.b_conv1
				self.h_pool1=max_pool_2x2(self.h_conv1)
				#self.h_drop1 = tf.nn.dropout(self.h_pool1, keep_prob=self.keep_prob)
				self.h_pool1_flat = tf.contrib.layers.flatten(self.h_pool1)

			with tf.variable_scope('linear_bottleneck'):
				self.W_fc0 = weight_variable([16*16*128,1200])
				self.b_fc0 = bias_variable([1200])
				self.h_fc0 = tf.matmul(self.h_pool1_flat, self.W_fc0) + self.b_fc0

			with tf.variable_scope('fc_layer_1'):
				self.W_fc1 = weight_variable([1200,1024*2])
				self.b_fc1 = bias_variable([1024*2])
				self.h_fc1 = tf.nn.relu(tf.matmul(self.h_fc0, self.W_fc1) + self.b_fc1)
				#self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)

			with tf.variable_scope('softmax'):
				self.W_fc2 = weight_variable([1024*2, 10])
				self.b_fc2 = bias_variable([10])
				self.logits = tf.matmul(self.h_fc1,self.W_fc2) + self.b_fc2
				self.prediction = tf.nn.softmax(self.logits)

			with tf.variable_scope('train'):
				self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_input-self.logits),axis=1),axis=0)
				self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def predict(self, x):
		prediction = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return prediction

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train, logits):
		train, l = self.sess.run([self.train_step, self.loss],\
	 		feed_dict = {self.X_input: X_train, self.y_input: logits, self.keep_prob: 1.0})
		prediction = self.predict(X_train)
		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/lsn/lsn.ckpt')
        	print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/lsn/lsn.ckpt')
        	print "Model Saved"

class DSN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'dsn' in v.name])

		if not os.path.exists("saved_model/dsn"):
			os.makedirs("saved_model/dsn")

		checkpoint = tf.train.get_checkpoint_state("saved_model/dsn/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.variable_scope('dsn'):
			with tf.variable_scope('inputs'):
				self.X_input = tf.placeholder('float32', [None,32,32,3])
				self.y_input = tf.placeholder('float32', [None, 10])
				self.keep_prob = tf.placeholder(tf.float32)

		    	with tf.variable_scope('conv_layer_1'):
                		self.W_conv1=weight_variable([3,3,3,48])
                		self.b_conv1=bias_variable([48])
                		self.h_conv1=conv2d(self.X_input,self.W_conv1)+self.b_conv1

            		with tf.variable_scope('conv_layer_2'):
                		self.W_conv2=weight_variable([3,3,48,96])
                		self.b_conv2=bias_variable([96])
                		self.h_conv2=conv2d(self.h_conv1,self.W_conv2)+self.b_conv2
                		self.h_pool2=max_pool_2x2(self.h_conv2)
                		self.h_pool2_flat = tf.contrib.layers.flatten(self.h_pool2)

            		with tf.variable_scope('linear_bottleneck'):
                		self.W_fc0 = weight_variable([16*16*96,1200])
                		self.b_fc0 = bias_variable([1200])
                		self.h_fc0 = tf.matmul(self.h_pool2_flat, self.W_fc0) + self.b_fc0

			with tf.variable_scope('fc_layer_1'):
				self.W_fc1 = weight_variable([1200,1024*2])
				self.b_fc1 = bias_variable([1024*2])
				self.h_fc1 = tf.nn.relu(tf.matmul(self.h_fc0, self.W_fc1) + self.b_fc1)
				#self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)

			with tf.variable_scope('softmax'):
				self.W_fc2 = weight_variable([1024*2, 10])
				self.b_fc2 = bias_variable([10])
				self.logits = tf.matmul(self.h_fc1,self.W_fc2) + self.b_fc2
				self.prediction = tf.nn.softmax(self.logits)

			with tf.variable_scope('train'):
				self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_input, logits = self.logits/temperature))
				self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def predict(self, x):
		prediction = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return prediction

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train, pred):
		train, l = self.sess.run([self.train_step, self.loss],\
	 		feed_dict = {self.X_input: X_train, self.y_input: pred, self.keep_prob: 1.0})
		prediction = self.predict(X_train)
		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/dsn/dsn.ckpt')
        	print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/dsn/dsn.ckpt')
        	print "Model Saved"

class MSN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'msn' in v.name])

		if not os.path.exists("saved_model/msn"):
			os.makedirs("saved_model/msn")

		checkpoint = tf.train.get_checkpoint_state("saved_model/msn/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv1d(x,W,stride):
			return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='VALID')

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.variable_scope('msn'):
			with tf.variable_scope('inputs'):
                		self.X_input = tf.placeholder('float32', [None,32,32,3])
                		self.y_input = tf.placeholder('float32', [None, 160])
                		self.keep_prob = tf.placeholder(tf.float32)

		        with tf.variable_scope('conv_layer_1'):
        				self.W_conv1=weight_variable([3,3,3,48])
        				self.b_conv1=bias_variable([48])
        				self.h_conv1=conv2d(self.X_input,self.W_conv1)+self.b_conv1

                	with tf.variable_scope('conv_layer_2'):
        				self.W_conv2=weight_variable([3,3,48,96])
        				self.b_conv2=bias_variable([96])
        				self.h_conv2=conv2d(self.h_conv1,self.W_conv2)+self.b_conv2
        				self.h_pool2=max_pool_2x2(self.h_conv2)
        				self.h_pool2_flat = tf.contrib.layers.flatten(self.h_pool2)

        		with tf.variable_scope('linear_bottleneck'):
        				self.W_fc0 = weight_variable([16*16*96,1200])
        				self.b_fc0 = bias_variable([1200])
        				self.h_fc0 = tf.matmul(self.h_pool2_flat, self.W_fc0) + self.b_fc0

        		with tf.variable_scope('fc_layer_1'):
        				self.W_fc1 = weight_variable([1200,1024])
        				self.b_fc1 = bias_variable([1024])
        				self.h_fc1 = tf.nn.relu(tf.matmul(self.h_fc0, self.W_fc1) + self.b_fc1)
                	with tf.variable_scope('fc_layer_2'):
        				self.W_fc2 = weight_variable([1024,1024])
        				self.b_fc2 = bias_variable([1024])
        				self.h_fc2 = tf.nn.sigmoid(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
			with tf.variable_scope('softmax'):
					self.W_fc2 = weight_variable([1024,160])
					self.b_fc2 = bias_variable([160])
					self.sublogits = tf.matmul(self.h_fc2, self.W_fc2) + self.b_fc2
					self.subpred = tf.nn.sigmoid(self.sublogits)
					self.logits = tf.reduce_sum(tf.reshape(self.subpred, [tf.shape(self.subpred)[0], 10, 16]),axis=2)
					self.prediction = tf.nn.softmax(self.logits)

			with tf.variable_scope('train'):
					self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y_input, logits = self.sublogits),axis=1),axis=0)
					self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def predict(self, x):
		pred = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		#pred = subpred.reshape(subpred.shape[0], 10, 4)
		#pred = np.sum(pred, axis=2)

		return pred

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train, subpred):
		train, l = self.sess.run([self.train_step, self.loss],\
	 		feed_dict = {self.X_input: X_train, self.y_input: subpred, self.keep_prob: 1.0})
		prediction = self.predict(X_train)
		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/msn/msn.ckpt')
        	print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/msn/msn.ckpt')
        	print "Model Saved"


class LNIN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		#self.saver = tf.train.Saver()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'lnin' in v.name])
		if not os.path.exists("saved_model/lnin"):
			os.makedirs("saved_model/lnin")

		checkpoint = tf.train.get_checkpoint_state("saved_model/lnin/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def conv1d(x,W,stride):
			return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='VALID')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.device('/gpu:0'):
			with tf.variable_scope('lnin'):
				with tf.variable_scope('inputs'):
					self.X_input = tf.placeholder('float32', [None,32,32,3])
					self.y_input = tf.placeholder('float32', [None, self.y_dims])
					self.keep_prob = tf.placeholder(tf.float32)

				with tf.variable_scope('conv_layer_1'):
					self.W_conv1=weight_variable([3,3,3,48])
					self.b_conv1=bias_variable([48])
					self.h_conv1=tf.nn.relu(conv2d(self.X_input,self.W_conv1)+self.b_conv1)

				with tf.variable_scope('conv_layer_2'):
					self.W_conv2=weight_variable([3,3,48,48])
					self.b_conv2=bias_variable([48])
					self.h_conv2=tf.nn.relu(conv2d(self.h_conv1,self.W_conv2)+self.b_conv2)
					self.h_pool2=max_pool_2x2(self.h_conv2)
					self.h_drop2 = tf.nn.dropout(self.h_pool2, keep_prob=self.keep_prob)

				with tf.variable_scope('conv_layer_3'):
					self.W_conv3=weight_variable([3,3,48,96])
					self.b_conv3=bias_variable([96])
					self.h_conv3=tf.nn.relu(conv2d(self.h_drop2,self.W_conv3)+self.b_conv3)

				with tf.variable_scope('conv_layer_4'):
					self.W_conv4=weight_variable([3,3,96,96])
					self.b_conv4=bias_variable([96])
					self.h_conv4=tf.nn.relu(conv2d(self.h_conv3,self.W_conv4)+self.b_conv4)
					self.h_pool4=max_pool_2x2(self.h_conv4)
					self.h_drop4 = tf.nn.dropout(self.h_pool4, keep_prob=self.keep_prob)


				with tf.variable_scope('conv_layer_5'):
					self.W_conv5=weight_variable([3,3,96,192])
					self.b_conv5=bias_variable([192])
					self.h_conv5=tf.nn.relu(conv2d(self.h_drop4,self.W_conv5)+self.b_conv5)

				with tf.variable_scope('conv_layer_6'):
					self.W_conv6=weight_variable([3,3,192,192])
					self.b_conv6=bias_variable([192])
					self.h_conv6=tf.nn.relu(conv2d(self.h_conv5,self.W_conv6)+self.b_conv6)
					self.h_pool6=max_pool_2x2(self.h_conv6)
					self.h_drop6 = tf.nn.dropout(self.h_pool6, keep_prob=self.keep_prob)
                	#flatten
					self.h_drop6_flat = tf.contrib.layers.flatten(self.h_drop6)


				with tf.variable_scope('fc_layer_1'):
					self.W_fc1 = weight_variable([4*4*192,512])
					self.b_fc1 = bias_variable([512])
					self.h_fc1 = tf.nn.relu(tf.matmul(self.h_drop6_flat, self.W_fc1) + self.b_fc1)
					self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)

				with tf.variable_scope('fc_layer_2'):
					self.h_conv1_flat = tf.contrib.layers.flatten(self.h_conv1)
					self.W_fc2 = weight_variable([512,256])
					self.b_fc2 = bias_variable([256])
					self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop,self.W_fc2) + self.b_fc2)
					self.h_fc2_drop = tf.nn.dropout(self.h_fc2, keep_prob=self.keep_prob)

				with tf.variable_scope('softmax'):
					self.W_fc3 = weight_variable([256, 10])
					self.b_fc3 = bias_variable([10])
					self.logits = tf.matmul(self.h_fc2_drop,self.W_fc3) + self.b_fc3
					self.prediction = tf.nn.softmax(self.logits)

				with tf.variable_scope('train'):
					self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.logits))
					self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
    	def get_hidden(self, x):
    		h = self.sess.run(self.h_fc2_drop, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
    		return h

    	def get_conv(self, x):
            	conv = self.sess.run(self.h_conv5, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
        	return conv

    	def correct(self, X, y):
        	pred = self.predict(X)
        	X = X[np.argmax(pred, axis=1) == np.argmax(y, axis=1)]
        	y = y[np.argmax(pred, axis=1) == np.argmax(y, axis=1)]
        	return X, y
	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def predict(self, x):
		pred = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return pred

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train):
		train, l, prediction = self.sess.run([self.train_step, self.loss, self.prediction],\
		 feed_dict = {self.X_input: X_train, self.y_input: y_train, self.keep_prob: .5})

		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/lnin/lnin.ckpt')
	        print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/lnin/lnin.ckpt')
	        print "Model Saved"

class DNIN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		#self.saver = tf.train.Saver()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'dnin' in v.name])
		if not os.path.exists("saved_model/dnin"):
			os.makedirs("saved_model/dnin")

		checkpoint = tf.train.get_checkpoint_state("saved_model/dnin/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def conv1d(x,W,stride):
			return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='VALID')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.device('/gpu:0'):
			with tf.variable_scope('dnin'):
				with tf.variable_scope('inputs'):
					self.X_input = tf.placeholder('float32', [None,32,32,3])
					self.y_input = tf.placeholder('float32', [None, self.y_dims])
					self.keep_prob = tf.placeholder(tf.float32)

				with tf.variable_scope('conv_layer_1'):
					self.W_conv1=weight_variable([3,3,3,48])
					self.b_conv1=bias_variable([48])
					self.h_conv1=tf.nn.relu(conv2d(self.X_input,self.W_conv1)+self.b_conv1)

				with tf.variable_scope('conv_layer_2'):
					self.W_conv2=weight_variable([3,3,48,48])
					self.b_conv2=bias_variable([48])
					self.h_conv2=tf.nn.relu(conv2d(self.h_conv1,self.W_conv2)+self.b_conv2)
					self.h_pool2=max_pool_2x2(self.h_conv2)
					self.h_drop2 = tf.nn.dropout(self.h_pool2, keep_prob=self.keep_prob)

				with tf.variable_scope('conv_layer_3'):
					self.W_conv3=weight_variable([3,3,48,96])
					self.b_conv3=bias_variable([96])
					self.h_conv3=tf.nn.relu(conv2d(self.h_drop2,self.W_conv3)+self.b_conv3)

				with tf.variable_scope('conv_layer_4'):
					self.W_conv4=weight_variable([3,3,96,96])
					self.b_conv4=bias_variable([96])
					self.h_conv4=tf.nn.relu(conv2d(self.h_conv3,self.W_conv4)+self.b_conv4)
					self.h_pool4=max_pool_2x2(self.h_conv4)
					self.h_drop4 = tf.nn.dropout(self.h_pool4, keep_prob=self.keep_prob)


				with tf.variable_scope('conv_layer_5'):
					self.W_conv5=weight_variable([3,3,96,192])
					self.b_conv5=bias_variable([192])
					self.h_conv5=tf.nn.relu(conv2d(self.h_drop4,self.W_conv5)+self.b_conv5)

				with tf.variable_scope('conv_layer_6'):
					self.W_conv6=weight_variable([3,3,192,192])
					self.b_conv6=bias_variable([192])
					self.h_conv6=tf.nn.relu(conv2d(self.h_conv5,self.W_conv6)+self.b_conv6)
					self.h_pool6=max_pool_2x2(self.h_conv6)
					self.h_drop6 = tf.nn.dropout(self.h_pool6, keep_prob=self.keep_prob)
                	#flatten
					self.h_drop6_flat = tf.contrib.layers.flatten(self.h_drop6)


				with tf.variable_scope('fc_layer_1'):
					self.W_fc1 = weight_variable([4*4*192,512])
					self.b_fc1 = bias_variable([512])
					self.h_fc1 = tf.nn.relu(tf.matmul(self.h_drop6_flat, self.W_fc1) + self.b_fc1)
					self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)

				with tf.variable_scope('fc_layer_2'):
					self.h_conv1_flat = tf.contrib.layers.flatten(self.h_conv1)
					self.W_fc2 = weight_variable([512,256])
					self.b_fc2 = bias_variable([256])
					self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop,self.W_fc2) + self.b_fc2)
					self.h_fc2_drop = tf.nn.dropout(self.h_fc2, keep_prob=self.keep_prob)

				with tf.variable_scope('softmax'):
					self.W_fc3 = weight_variable([256, 10])
					self.b_fc3 = bias_variable([10])
					self.logits = tf.matmul(self.h_fc2_drop,self.W_fc3) + self.b_fc3
					self.prediction = tf.nn.softmax(self.logits/temperature)

				with tf.variable_scope('train'):
					self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.logits/temperature))
					self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
    	def get_hidden(self, x):
    		h = self.sess.run(self.h_fc2_drop, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
    		return h

    	def get_conv(self, x):
            	conv = self.sess.run(self.h_conv5, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
        	return conv

    	def correct(self, X, y):
        	pred = self.predict(X)
        	X = X[np.argmax(pred, axis=1) == np.argmax(y, axis=1)]
        	y = y[np.argmax(pred, axis=1) == np.argmax(y, axis=1)]
        	return X, y
	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def predict(self, x):
		pred = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return pred

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train):
		train, l, prediction = self.sess.run([self.train_step, self.loss, self.prediction],\
		 feed_dict = {self.X_input: X_train, self.y_input: y_train, self.keep_prob: .5})

		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/dnin/dnin.ckpt')
	        print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/dnin/dnin.ckpt')
	        print "Model Saved"

class MNIN(object):
	def __init__(self, x_dims, y_dims, sess, choice):
		self.sess = sess
		self.x_dims = x_dims
		self.y_dims = y_dims
		self.choice = choice
		self.build_model()
		#self.saver = tf.train.Saver()
		self.saver = tf.train.Saver([v for v in tf.all_variables() if 'mnin' in v.name])
		if not os.path.exists("saved_model/mnin"):
			os.makedirs("saved_model/mnin")

		checkpoint = tf.train.get_checkpoint_state("saved_model/mnin/")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(sess, checkpoint.model_checkpoint_path)
			print "Model Restored:", checkpoint.model_checkpoint_path
			self.save = True
		else:
			print "Could not find old network weights"
			self.save = False

	def build_model(self):
		def weight_variable(shape):
			W = tf.get_variable('W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
			return W

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		def conv2d(x,W):
			return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

		def conv1d(x,W,s1,s2):
			return tf.nn.conv2d(x,W,strides=[1,s1,s2,1], padding='VALID')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

		with tf.device('/gpu:0'):
			with tf.variable_scope('mnin'):
				with tf.variable_scope('inputs'):
					self.X_input = tf.placeholder('float32', [None,32,32,3])
					self.y_input = tf.placeholder('float32', [None, self.y_dims])
					self.keep_prob = tf.placeholder(tf.float32)

				with tf.variable_scope('conv_layer_1'):
					self.W_conv1=weight_variable([3,3,3,48])
					self.b_conv1=bias_variable([48])
					self.h_conv1=tf.nn.relu(conv2d(self.X_input,self.W_conv1)+self.b_conv1)

				with tf.variable_scope('conv_layer_2'):
					self.W_conv2=weight_variable([3,3,48,48])
					self.b_conv2=bias_variable([48])
					self.h_conv2=tf.nn.relu(conv2d(self.h_conv1,self.W_conv2)+self.b_conv2)
					self.h_pool2=max_pool_2x2(self.h_conv2)
					self.h_drop2 = tf.nn.dropout(self.h_pool2, keep_prob=self.keep_prob)

				with tf.variable_scope('conv_layer_3'):
					self.W_conv3=weight_variable([3,3,48,96])
					self.b_conv3=bias_variable([96])
					self.h_conv3=tf.nn.relu(conv2d(self.h_drop2,self.W_conv3)+self.b_conv3)

				with tf.variable_scope('conv_layer_4'):
					self.W_conv4=weight_variable([3,3,96,96])
					self.b_conv4=bias_variable([96])
					self.h_conv4=tf.nn.relu(conv2d(self.h_conv3,self.W_conv4)+self.b_conv4)
					self.h_pool4=max_pool_2x2(self.h_conv4)
					self.h_drop4 = tf.nn.dropout(self.h_pool4, keep_prob=self.keep_prob)


				with tf.variable_scope('conv_layer_5'):
					self.W_conv5=weight_variable([3,3,96,192])
					self.b_conv5=bias_variable([192])
					self.h_conv5=tf.nn.relu(conv2d(self.h_drop4,self.W_conv5)+self.b_conv5)

				with tf.variable_scope('conv_layer_6'):
					self.W_conv6=weight_variable([3,3,192,192])
					self.b_conv6=bias_variable([192])
					self.h_conv6=tf.nn.relu(conv2d(self.h_conv5,self.W_conv6)+self.b_conv6)
					self.h_pool6=max_pool_2x2(self.h_conv6)
					self.h_drop6 = tf.nn.dropout(self.h_pool6, keep_prob=self.keep_prob)
                	#flatten
					self.h_drop6_flat = tf.contrib.layers.flatten(self.h_drop6)


				with tf.variable_scope('fc_layer_1'):
					self.W_fc1 = weight_variable([8*8*96,512])
					self.b_fc1 = bias_variable([512])
					self.h_fc1 = tf.nn.relu(tf.matmul(self.h_drop6_flat, self.W_fc1) + self.b_fc1)
					self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=self.keep_prob)

				with tf.variable_scope('fc_layer_2'):
					self.h_conv1_flat = tf.contrib.layers.flatten(self.h_conv1)
					self.W_fc2 = weight_variable([512,256])
					self.b_fc2 = bias_variable([256])
					self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop,self.W_fc2) + self.b_fc2)
					self.h_fc2_drop = tf.nn.dropout(self.h_fc2, keep_prob=self.keep_prob)
                    			self.h_fc2_reshape = tf.reshape(self.h_fc2_drop, [tf.shape(self.h_fc2_drop)[0],16,16,1])

				with tf.variable_scope('conv_layer_7'):
					self.W_conv7=weight_variable([4,4,1,10])
					self.b_conv7=bias_variable([10])
					self.sublogits = conv1d(self.h_fc2_reshape,self.W_conv7,4,4) + self.b_conv7
					self.subpred = tf.nn.sigmoid(tf.reshape(self.sublogits, [tf.shape(self.sublogits)[0], 16, 10]))

                		with tf.variable_scope('decoder_1'):
                    			self.W_dc1=weight_variable([256, 32*32*3])
					self.b_dc1=bias_variable([32*32*3])
                    			self.recon=tf.nn.sigmoid(tf.matmul(self.h_fc2_drop, self.W_dc1)+self.b_dc1)
                                self.recon_target = tf.reshape(self.X_input, [tf.shape(self.X_input)[0], 32*32*3])
                    		"""
                		with tf.variable_scope('decoder_2'):
                    			self.W_dc2=weight_variable([512, 512])
					self.b_dc2=bias_variable([512])
                    			self.h_dc2=tf.nn.relu(tf.matmul(self.h_dc1, self.W_dc2)+self.b_dc2)

                		with tf.variable_scope('decoder_3'):
                    			self.W_dc3=weight_variable([512, 32*32*3])
					self.b_dc3=bias_variable([32*32*3])
                    			self.recon=tf.nn.sigmoid(tf.matmul(self.h_dc2, self.W_dc3)+self.b_dc3)
                    			self.recon_target = tf.reshape(self.X_input, [tf.shape(self.X_input)[0], 32*32*3])
                    		"""
				with tf.variable_scope('softmax'):
					self.logits = tf.reduce_sum(self.subpred,axis=1)
					self.prediction = tf.nn.softmax(self.logits)
                    			self.pj = tf.reduce_mean(self.subpred, axis=0)

				with tf.variable_scope('train'):
					self.categorical_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.logits))
                    			self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.recon-self.recon_target),axis=1),axis=0)
                    			self.sparse_loss = tf.reduce_sum(p*tf.log(p/self.pj)+(1-p)*tf.log((1-p)/(1-self.pj)))
                    			self.reg_loss = tf.reduce_sum(tf.square(self.W_dc1))
                    			self.loss = self.categorical_loss+C*self.recon_loss+BETA*self.sparse_loss+LAMBDA*self.reg_loss
                    			self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def get_sub_logits(self, x):
		sublogits = self.sess.run(self.sublogits_T, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return sublogits

	def get_logits(self, x):
		logits = self.sess.run(self.logits, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return logits

	def get_sub_prediction(self, x):
		subpred = self.sess.run(self.subpred, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
        	subpred = np.swapaxes(subpred,1,2)
        	subpred = subpred.reshape(x.shape[0],160)
		return subpred

	def predict(self, x):
		pred = self.sess.run(self.prediction, feed_dict = {self.X_input: x, self.keep_prob: 1.0})
		return pred

	def accuracy(self, pred, y):
		correct = 0
		for i in range(pred.shape[0]):
			if np.argmax(pred[i]) == np.argmax(y[i]):
				correct+=1

		accuracy = 1.*correct/pred.shape[0]

		return accuracy

	def test(self, X_test, y_test):
		pred = self.predict(X_test)
		accuracy = self.accuracy(pred, y_test)

		return accuracy

	def train(self, X_train, y_train):
		train, l, prediction = self.sess.run([self.train_step, self.loss, self.prediction],\
		 feed_dict = {self.X_input: X_train, self.y_input: y_train, self.keep_prob: .5})

		a = self.accuracy(prediction, y_train)
		return l, a

	def restore_model(self):
		self.saver.restore(self.sess, 'saved_model/mnin/mnin.ckpt')
	        print "Model Restored"

	def save_model(self):
		self.saver.save(self.sess, 'saved_model/mnin/mnin.ckpt')
	        print "Model Saved"
