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
    eps_nbasis = config['eps_nbasis']

    # Center x at 0
    x = 2*x - 1

    # Noise
    if eps is None:
        batch_size = tf.shape(x)[0]
        eps = tf.random_normal(tf.stack([eps_nbasis, batch_size, eps_dim]))

    net = flatten_spatial(x)
    net = slim.fully_connected(net, 768, activation_fn=tf.nn.softplus, scope="fc_0")

    z0 = slim.fully_connected(net, z_dim, activation_fn=None, scope='z0',
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-5))

    a_vec = []
    for i in range(eps_nbasis):
        a = slim.fully_connected(net, z_dim, activation_fn=None, scope='a_%d' % i)
        a = tf.nn.elu(a - 5.) + 1.
        a_vec.append(a)

    # Noise basis
    v_vec = []
    for i in range(eps_nbasis):
        with tf.variable_scope("eps_%d" % i):
            fc_argscope = slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.elu)
            with fc_argscope:
                net = slim.fully_connected(eps[i], 128, scope='fc_0')
                net = slim.fully_connected(net, 128, scope='fc_1')
                net = slim.fully_connected(net, 128, scope='fc_2')
            v = slim.fully_connected(net, z_dim, activation_fn=None, scope='v')

            v_vec.append(v)

    # Sample and Moments
    z = z0
    Ez = z0
    Varz = 0.

    for a, v in zip(a_vec, v_vec):
        z += a*v
        Ev, Varv = tf.nn.moments(v, [0])
        Ez += a*Ev
        Varz += a*a*Varv

    # if z_dist == "uniform":
    #     z = tf.nn.sigmoid(z)

    return z, Ez, Varz
