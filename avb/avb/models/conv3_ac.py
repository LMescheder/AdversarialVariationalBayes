import tensorflow as tf
from tensorflow.contrib import slim
from avb.ops import *

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

    conv2dtrp_argscope = slim.arg_scope([conv2d_transpose],
                            activation_fn=None, kernel_size=(3,3), stride=(2, 2))
    conv2d_argscope = slim.arg_scope([slim.conv2d],
            activation_fn=None, kernel_size=(3,3), stride=1)

    conv2d_argscope = slim.arg_scope([slim.conv2d],
            activation_fn=None, kernel_size=(3,3), stride=1)

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
