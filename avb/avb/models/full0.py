import tensorflow as tf
from tensorflow.contrib import slim as slim
from avb.ops import *
import math

def encoder(x, config, eps=None, is_training=True):
    output_size = config['output_size']
    c_dim = config['c_dim']
    df_dim = config['df_dim']
    z_dist = config['z_dist']
    z_dim = config['z_dim']
    eps_dim = config['eps_dim']

    # Center x at 0
    x = 2*x - 1

    # Noise
    if eps is None:
        batch_size = tf.shape(x)[0]
        eps = tf.random_normal(tf.stack([batch_size, eps_dim]))

    net = flatten_spatial(x)
    net = add_linear(eps, net)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus, scope="fc_0")
    net = add_linear(eps, net)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus, scope="fc_1")
    net = add_linear(eps, net)

    z = slim.fully_connected(net, z_dim, activation_fn=None, scope='z',
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-5))

    if z_dist == "uniform":
        z = tf.nn.sigmoid(z)

    return z
