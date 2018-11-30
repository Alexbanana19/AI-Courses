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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_size = 32*32*3
img_class = 10
batch_size = 128
channels = 80
def model_loss(y, logits, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out


def fgsm(x, predictions, y, eps=0.3, clip_min=None, clip_max=None):
    return fgm(x, predictions, y, eps=eps, ord=np.inf, clip_min=clip_min,
               clip_max=clip_max)


def fgm(x, preds, y=None, eps=0.3, ord=np.inf,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    #y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = np.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        #normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(range(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    print type(normalized_grad)
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

X = np.load("X.npy")
y = np.load("y.npy")
r = 0

X_test = X[r]
y_test = y[r]

X_train = np.delete(X, r, 0).reshape(50000,32,32,3)
y_train = np.delete(y, r, 0).reshape(50000,10)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    model = DNIN(img_size, img_class, sess, 5)
    logits = model.get_logits(X_train[0:100])
    X_adv = fgsm(X_train[0:100], logits, y_train[0:100])
    for i in range(1,500):
        print "batch ", i
        logits = model.get_logits(X_train[i*100:(i+1)*100])
        X_adv = np.concatenate((X_adv, fgsm(X_train[i*100:(i+1)*100], logits, y_train[i*100:(i+1)*100])), axis = 0)

np.save("X_adv.npy",X_adv)
