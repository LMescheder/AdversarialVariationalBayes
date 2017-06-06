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

    # Network
    net = slim.fully_connected(z, 768, activation_fn=tf.nn.softplus, scope="fc_0")
    output = [
        slim.fully_connected(net, s*s*c_dim, activation_fn=None, scope="x_%i" % i) for i in range(num_out)
    ]
    output = [tf.reshape(x, [-1, s, s, c_dim]) for x in output]

    return output
