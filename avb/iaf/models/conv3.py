import tensorflow as tf
from tensorflow.contrib import slim
from avb.ops import *
import numpy as np
import ipdb

def encoder(x, config, is_training=True):
    df_dim = config['df_dim']
    z_dim = config['z_dim']
    a_dim = config['iaf_a_dim']

    # Center x at 0
    x = 2*x - 1

    bn_kwargs = {
        'scale': True, 'center':True, 'is_training': is_training, 'updates_collections': None
    }

    conv2d_argscope = slim.arg_scope([slim.conv2d],
            activation_fn=None, kernel_size=(3,3), stride=1,
            normalizer_fn=None, normalizer_params=bn_kwargs)

    net = x
    with conv2d_argscope:
        # 2-strided resnet block
        res = slim.conv2d(net, 16, activation_fn=tf.nn.elu, stride=2)
        res = slim.conv2d(res, 16)
        net = tf.nn.elu(slim.conv2d(net, 16, stride=2) + res)

        # 1-strided resnet block
        net = slim.conv2d(net, 32)
        res = slim.conv2d(net, 32, activation_fn=tf.nn.elu)
        res = slim.conv2d(res, 32)
        net = tf.nn.elu(net + res)

        # 2-strided resnet block
        res = slim.conv2d(net, 32, activation_fn=tf.nn.elu, stride=2)
        res = slim.conv2d(res, 32)
        net = tf.nn.elu(slim.conv2d(net, 32, stride=2) + res)

        # 1-strided resnet block
        net = slim.conv2d(net, 32)
        res = slim.conv2d(net, 32, activation_fn=tf.nn.elu)
        res = slim.conv2d(res, 32)
        net = tf.nn.elu(net + res)

        # 2-strided resnet block
        res = slim.conv2d(net, 32, activation_fn=tf.nn.elu, stride=2)
        res = slim.conv2d(res, 32)
        net = tf.nn.elu(slim.conv2d(net, 32, stride=2) + res)

    net = flatten_spatial(net)
    net = slim.fully_connected(net, 450, activation_fn=tf.nn.elu)


    zmean = slim.fully_connected(net, z_dim, activation_fn=None)
    log_zstd = slim.fully_connected(net, z_dim, activation_fn=None)
    a = slim.fully_connected(net, a_dim, activation_fn=None)

    return zmean, log_zstd, a
