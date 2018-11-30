from __future__ import absolute_import
import numpy as np
import random
import tensorflow as tf
import pickle
import sys, os
sys.path.append('../')
sys.path.append('/home/share/minghan/ssae/')
sys.path.append('/home/share/minghan/keras/lib/python2.7/site-packages')

from argparse import ArgumentParser

from models import DNIN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

img_size = 32*32*3
img_class = 10
batch_size = 128
# mix the data
X = np.load("X.npy")
y = np.load("y.npy")
r = 0

X_test = X[r]
y_test = y[r]

X_train = np.delete(X, r, 0).reshape(50000,32,32,3)
y_train = np.delete(y, r, 0).reshape(50000,10)

X_train1 = X_train.copy()
y_train1 = y_train.copy()
s = np.arange(X_train1.shape[0])
np.random.shuffle(s)
X_train1 = X_train1[s]
y_train1 = y_train1[s]

X_train2 = X_train.copy()
y_train2 = y_train.copy()
s = np.arange(X_train2.shape[0])
np.random.shuffle(s)
X_train2 = X_train2[s]
y_train2 = y_train2[s]

p = np.random.beta(0.4,0.4,X_train1.shape[0])
p = np.tile(p[:,None,None,None],(1, 32, 32, 3))
X_train_mix = p*X_train1+(1-p)*X_train2
y_train_mix = y_train1

X_train_mix = np.concatenate((X_train,X_train_mix))
y_train_mix = np.concatenate((y_train,y_train_mix))

# get soft labels
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    teacher_model = DNIN(img_size, img_class, sess, choice)
    if not teacher_model.save:
        print "No model"

    soft_labels = teacher_model.predict(X_train_mix[0:batch_size])
    step = 1
    while step*batch_size < X_train_mix.shape[0]:
        X_batch = X_train_mix[step*batch_size:min(X_train_mix.shape[0],(step+1)*batch_size), :]

        labels = teacher_model.predict(X_batch)
        soft_labels = np.concatenate((soft_labels,labels))
        step += 1

    print X_train_mix.shape, np.save('X_train_mix.npy',X_train_mix)
    print soft_labels.shape, np.save('soft_labels.npy',soft_labels)
    print y_train_mix.shape, np.save('y_train_mix.npy',X_train_mix)
