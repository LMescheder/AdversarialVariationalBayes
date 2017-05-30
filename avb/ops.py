import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.contrib import slim
from tensorflow.contrib import layers as tflayers

from avb.utils import *

@slim.add_arg_scope
def conv2d_transpose(
        inputs,
        out_shape,
        kernel_size=(5, 5),
        stride=(1, 1),
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tflayers.xavier_initializer(),
        scope=None,
        reuse=None):
    batchsize = tf.shape(inputs)[0]
    in_channels = int(inputs.get_shape()[-1])

    output_shape = tf.stack([batchsize, out_shape[0], out_shape[1], out_shape[2]])
    filter_shape = [kernel_size[0], kernel_size[1], out_shape[2], in_channels]

    with tf.variable_scope(scope, 'Conv2d_transpose', [inputs], reuse=reuse) as sc:
        w = tf.get_variable('weights', filter_shape,
                            initializer=weights_initializer)

        outputs = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                            strides=[1, stride[0], stride[1], 1])

        if not normalizer_fn:
            biases = tf.get_variable('biases', [out_shape[2]], initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

@slim.add_arg_scope
def add_linear(
        inputs,
        targets,
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tflayers.xavier_initializer(),
        scope=None,
        reuse=None):
    with tf.variable_scope(scope, 'AddLinear', [inputs], reuse=reuse) as sc:
        shape_targets = targets.get_shape()
        targets_size = int(shape_targets[1]) * int(shape_targets[2]) * int(shape_targets[3])
        outputs = slim.fully_connected(inputs, targets_size, activation_fn=None, weights_initializer=weights_initializer)
        outputs = tf.reshape(outputs, tf.shape(targets))
        outputs = outputs + targets

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

@slim.add_arg_scope
def add_resnet_conv(
        inputs,
        channels,
        nlayers=1,
        kernel_size=(5, 5),
        activation_fn=tf.nn.elu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tflayers.xavier_initializer(),
        scope=None,
        reuse=None):
    with tf.variable_scope(scope, 'Resnet_conv', [inputs], reuse=reuse) as sc:
        channels_in = int(inputs.get_shape()[3])
        net = inputs
        with slim.arg_scope([slim.conv2d], kernel_size=kernel_size, stride=(1, 1)):
            for i in range(nlayers):
                net = activation_fn(net)
                res = slim.conv2d(net, channels,
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    scope="res_%d_0" % i)
                res = slim.conv2d(net, channels_in,
                    activation_fn=None, scope="res_%d_1" % i)
                net += res

    return net

@slim.add_arg_scope
def masked_linear_layer(
        inputs,
        out_dim,
        mask,
        activation_fn=None,
        weights_initializer=tflayers.xavier_initializer(),
        scope=None,
        reuse=None):
    with tf.variable_scope(scope, 'MADE', [inputs], reuse=reuse) as sc:
        in_dim = int(inputs.get_shape()[1])
        M = tf.constant(mask)
        W = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=weights_initializer)
        biases = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))

        out = tf.matmul(inputs, M*W) + biases

        if not activation_fn is None:
            out = activation_fn(out)

    return out

def custom_initializer(seed=None, dtype=tf.float32, trp=False):
    def _initializer(shape, dtype=dtype, partition_info=None):
        if len(shape) == 2:
            N = float(shape[1])
        elif len(shape) == 4 and not trp:
            N = float(shape[0]) * float(shape[1]) * float(shape[2])
        elif len(shape) == 4 and trp:
            N = float(shape[0]) * float(shape[1]) * float(shape[3])
        else:
            raise ValueError("weights need to be either 2 or 4!")
        stddev = 1./math.sqrt(N)
        return tf.truncated_normal(shape, 0.0, stddev, dtype, seed=seed)
    return _initializer

def flatten_spatial(x):
    x_shape = x.get_shape().as_list()
    x_dim = np.prod(x_shape[1:])
    x_flat = tf.reshape(x, [-1, x_dim])
    return x_flat

def norm(x, axes=None, keep_dims=False):
    return tf.sqrt(tf.reduce_sum(x*x, reduction_indices=axes, keep_dims=keep_dims))

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def reduce_geomean(x, axis=None):
    "Computes log(sum_i exp(x_i) / N)."
    N = tf.reduce_prod(tf.shape(x)[axis])
    out = tf.reduce_logsumexp(x, axis=axis) - tf.log(tf.to_float(N))
    return out

def tril_matrix(n, unit_diag=True):
    offset = 0
    nentries = n*(n+1)/2
    if unit_diag:
        offset = -1
        nentries = (n-1)*n/2

    indices = list(zip(*np.tril_indices(n, offset)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int32)

    weights = tf.get_variable('weights', [nentries], initializer=tf.constant_initializer(0.0))

    matrix = tf.sparse_to_dense(sparse_indices=indices, output_shape=[n, n],
        sparse_values=weights, default_value=0, validate_indices=True)

    if unit_diag:
        matrix += tf.constant(np.eye(n, dtype=np.float32))

    return matrix


def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        summaries = tf.summary.merge([
            tf.summary.scalar("mean", mean),
            tf.summary.scalar("stddev", stddev),
            # tf.scalar_summary("median/" + name, tf.reduce_median(var))
            tf.summary.histogram("hist", var),
        ])
        return summaries
