import tensorflow as tf
from tensorflow.contrib import slim
from avb.ops import *

def encoder(x, config, is_training=True):
    df_dim = config['df_dim']
    z_dim = config['z_dim']
    output_size = config['output_size']
    a_dim = config['iaf_a_dim']

    # Center x at 0
    x = 2*x - 1

    n_down = max(min(int(math.log(output_size, 2)) - 2, 4), 0)
    filter_strides = [(1, 1)] * (4 - n_down) + [(2, 2)] * n_down

    bn_kwargs = {
        'scale': True, 'center':True, 'is_training': is_training, 'updates_collections': None
    }

    conv2d_argscope = slim.arg_scope([slim.conv2d],
        activation_fn=tf.nn.elu, kernel_size=(5, 5),
        normalizer_fn=None, normalizer_params=bn_kwargs
    )

    with conv2d_argscope:
        net = slim.conv2d(x, 1*df_dim, stride=filter_strides[0], scope="conv_0")
        net = slim.conv2d(net, 2*df_dim, stride=filter_strides[1], scope="conv_1")
        net = slim.conv2d(net, 4*df_dim, stride=filter_strides[2], scope="conv_2")
        net = slim.conv2d(net, 8*df_dim, stride=filter_strides[3], normalizer_fn=None, scope="conv_3")

    net = flatten_spatial(net)

    zmean = slim.fully_connected(net, z_dim, activation_fn=None)
    log_zstd = slim.fully_connected(net, z_dim, activation_fn=None)
    a = slim.fully_connected(net, a_dim, activation_fn=None)

    return zmean, log_zstd, a
