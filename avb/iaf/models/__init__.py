import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from avb.iaf.models import (
    conv1
)
from avb.ops import masked_linear_layer

encoder_dict = {
    'conv1': conv1.encoder,
}


def get_encoder(model_name, config, scope='encoder'):
    model_func = encoder_dict[model_name]

    return tf.make_template(
        scope, model_func, config=config
    )

def get_iaf_layer(model_name, config, scope='iaf_layer'):
    h_dim = config['iaf_h_dim']
    z_dim = config['z_dim']

    return tf.make_template(
        scope, iaf_layer, config=config
    )

def iaf_layer(z_in, a_in, config, activation_fn=tf.nn.relu):
    h_dim = config['iaf_h_dim']
    z_dim = config['z_dim']

    # Masks
    m = np.random.randint(1, z_dim, size=h_dim)
    d = np.arange(1, z_dim+1)
    M1_np = (m.reshape(1, -1) >= d.reshape(-1, 1)).astype(np.float32)
    M2_np = (m.reshape(-1, 1) < d.reshape(1, -1)).astype(np.float32)
    M1 = tf.get_variable('M1', initializer=M1_np, trainable=False)
    M2 = tf.get_variable('M2', initializer=M2_np, trainable=False)

    # Network
    net = masked_linear_layer(z_in, h_dim, M1, activation_fn=None)
    net += slim.fully_connected(a_in, h_dim, activation_fn=None)
    net = activation_fn(net)

    m = 0.1 * masked_linear_layer(net, z_dim, M2, activation_fn=None)
    s = 2. + 0.1 * masked_linear_layer(net, z_dim, M2, activation_fn=None)

    return m, s
