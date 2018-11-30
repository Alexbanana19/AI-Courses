from __future__ import absolute_import
import sys, os
sys.path.append('../')
sys.path.append('/home/share/minghan/ssae/')
sys.path.append('/home/share/minghan/keras/lib/python2.7/site-packages')
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from models import MNIN
from keras.datasets import cifar10, mnist
from keras.utils.np_utils import to_categorical
import pickle

img_size = 32*32*3
#img_size = 28*28
img_class = 10
batch_size = 128
period = 1000

X = np.load("X.npy")
y = np.load("y.npy")
print X.shape
X = X[5]
np.save("X_5.npy", X)
"""

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    #model = Convnet(img_size, img_class, sess)
    model = MNIN(img_size, img_class, sess, 5)
    logits = model.get_sub_prediction(X[0:100])
    print logits.shape
    for i in range(1,100):
        print "batch ", i
        print model.get_sub_prediction(X[i*100:(i+1)*100]).shape
        logits= np.concatenate((logits, model.get_sub_prediction(X[i*100:(i+1)*100])), axis = 0)
    np.save("mnin_logits.npy",logits)"""
