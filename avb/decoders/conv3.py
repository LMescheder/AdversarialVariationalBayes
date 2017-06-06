import tensorflow as tf
from tensorflow.contrib import slim as slim
from avb.ops import *
import math

def decoder(z, config, num_out=1, is_training=True):
    output_size = config['output_size']
    c_dim = config['c_dim']
    gf_dim = config['gf_dim']

    # Layers in which downsampling occurs
    s = output_size
    s2, s4, s8 = int(np.ceil(s/2)), int(np.ceil(s/4)), int(np.ceil(s/8))

    # Network
    net = slim.fully_connected(z, 300, activation_fn=tf.nn.softplus, scope="fc_0")
    net = slim.fully_connected(net, s8 * s8 * 32, activation_fn=tf.nn.softplus, scope="fc_1")

    net = tf.reshape(net, [-1, s8, s8, 32])

    conv2dtrp_argscope = slim.arg_scope([conv2d_transpose],
                            activation_fn=tf.nn.softplus, kernel_size=(3,3), stride=(2, 2))
    with conv2dtrp_argscope:
        dnet = slim.conv2d(net, 32, kernel_size=(3, 3), activation_fn=tf.nn.softplus)
        net = tf.nn.softplus(net + dnet)
        net = conv2d_transpose(net, [s4, s4, 32], scope="conv_0")

        dnet = slim.conv2d(net, 32, kernel_size=(3, 3), activation_fn=tf.nn.softplus)
        net = tf.nn.softplus(net + dnet)
        net = conv2d_transpose(net, [s2, s2, 16], scope="conv_1")

        dnet = slim.conv2d(net, 16, kernel_size=(3, 3), activation_fn=tf.nn.softplus)
        net = tf.nn.softplus(net + dnet)
        output = [
            conv2d_transpose(net, [s, s, c_dim], activation_fn=None, scope="x_%i" % i)
            for i in range(num_out)
        ]

    return output
