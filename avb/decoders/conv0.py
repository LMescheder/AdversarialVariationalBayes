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
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    n_up = max(min(int(math.log(output_size, 2)) - 2, 4), 0)
    filter_strides = [(1, 1)] * (4 - n_up) + [(2, 2)] * n_up
    s_down = [s] * (4 - n_up) + [s2, s4, s8, s16][:n_up]

    # Network
    net = slim.fully_connected(z, gf_dim*8*s_down[3]*s_down[3], activation_fn=tf.nn.elu, scope="fc_0")
    net = tf.reshape(net, [-1, s_down[3], s_down[3], gf_dim*8])
    with slim.arg_scope([slim.conv2d_transpose],
            activation_fn=tf.nn.elu, kernel_size=(5, 5)):
        net = conv2d_transpose(net, [s_down[2], s_down[2], 4*gf_dim], stride=filter_strides[3], scope="conv_0")
        net = conv2d_transpose(net, [s_down[1], s_down[1], 2*gf_dim], stride=filter_strides[2], scope="conv_1")
        net = conv2d_transpose(net, [s_down[0], s_down[0], 1*gf_dim], stride=filter_strides[1], scope="conv_2")


    output = [
        conv2d_transpose(net, [s, s, c_dim], stride=filter_strides[0],
            activation_fn=None, kernel_size=(5, 5), scope="x_%i" % i)
        for i in range(num_out)
    ]

    return output
