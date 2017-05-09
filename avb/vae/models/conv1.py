import tensorflow as tf
from tensorflow.contrib import slim
from avb.ops import *

def encoder(x, config, is_training=True):
    df_dim = config['df_dim']
    z_dim = config['z_dim']

    # Center x at 0
    x = 2*x - 1

    bn_kwargs = {
        'scale': True, 'center':True, 'is_training': is_training, 'updates_collections': None
    }

    conv2d_argscope = slim.arg_scope([slim.conv2d],
            activation_fn=tf.nn.softplus, kernel_size=(5, 5), stride=2,
            normalizer_fn=None, normalizer_params=bn_kwargs)

    with conv2d_argscope:
        net = slim.conv2d(x, 16, scope="conv_0")
        net = slim.conv2d(net, 32, scope="conv_1")
        net = slim.conv2d(net, 32, scope="conv_2", normalizer_fn=None)

    net = flatten_spatial(net)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.softplus, scope='fc_0')


    zmean = slim.fully_connected(net, z_dim, activation_fn=None)
    log_zstd = slim.fully_connected(net, z_dim, activation_fn=None)

    return zmean, log_zstd
