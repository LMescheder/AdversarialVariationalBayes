import tensorflow as tf
from tensorflow.contrib import slim as slim
from avb.ops import *
import math


def encoder(x, a, config, is_training=True):
    z_dim = config['z_dim']

    # Center x at 0
    x = 2*x - 1
    x = flatten_spatial(x)

    net = tf.concat([x, a], axis=1)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus)

    zmean = slim.fully_connected(net, z_dim, activation_fn=None)
    log_zstd = slim.fully_connected(net, z_dim, activation_fn=None)

    return zmean, log_zstd


def encoder_aux(x, config, is_training=True):
    a_dim = config['a_dim']

    # Center x at 0
    x = 2*x - 1

    net =  flatten_spatial(x)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus)

    amean = slim.fully_connected(net, a_dim, activation_fn=None)
    log_astd = slim.fully_connected(net, a_dim, activation_fn=None)

    return amean, log_astd

def decoder_aux(x, z, config, is_training=True):
    a_dim = config['a_dim']

    x = 2*x - 1

    x = flatten_spatial(x)
    net = tf.concat([x, z], axis=1)

    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus)


    amean = slim.fully_connected(net, a_dim, activation_fn=None)
    log_astd = slim.fully_connected(net, a_dim, activation_fn=None)

    return amean, log_astd
