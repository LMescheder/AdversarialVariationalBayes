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
    a_dim = config['iaf_a_dim']
    z_dim = config['z_dim']

    m = np.random.randint(1, z_dim-1, size=a_dim)

    return tf.make_template(
        scope, iaf_layer, m=m, config=config
    )


def iaf_layer(z_in, a_in, m, config, activation_fn=tf.nn.relu):
    a_dim = config['iaf_a_dim']
    z_dim = config['z_dim']

    d = np.arange(z_dim)
    M1 = (m.reshape(1, -1) >= d.reshape(-1, 1)).astype(np.float32)
    M2 = (m.reshape(-1, 1) < d.reshape(1, -1)).astype(np.float32)

    net = masked_linear_layer(z_in, a_dim, M1, activation_fn=None)
    net += slim.fully_connected(a_in, a_dim, activation_fn=None)
    net = activation_fn(net)

    m = 0.1 * masked_linear_layer(net, z_dim, M2, activation_fn=None)
    s = 0.1 * masked_linear_layer(net, z_dim, M2, activation_fn=None)

    return m, s
