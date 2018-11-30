import numpy as np
import tensorflow as tf
import os

batch_size = 128
temperature = 1.0

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
