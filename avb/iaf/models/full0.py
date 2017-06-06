import tensorflow as tf
from tensorflow.contrib import slim as slim
from avb.ops import *
import math

def encoder(x, config, is_training=True):
    df_dim = config['df_dim']
    z_dim = config['z_dim']
    a_dim = config['iaf_a_dim']

    # Center x at 0
    x = 2*x - 1

    net = flatten_spatial(x)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus, scope="fc_0")
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus, scope="fc_1")

    zmean = slim.fully_connected(net, z_dim, activation_fn=None)
    log_zstd = slim.fully_connected(net, z_dim, activation_fn=None)
    a = slim.fully_connected(net, a_dim, activation_fn=None)

    return zmean, log_zstd, a
