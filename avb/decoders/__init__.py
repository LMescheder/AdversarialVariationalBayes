import tensorflow as tf
import numpy as np
from avb.decoders import (
    conv0, conv1, conv2,
)

decoder_dict = {
    'conv0': conv0.decoder,
    'conv1': conv1.decoder,
    'conv2': conv2.decoder,
}

def get_decoder(model_name, config, scope='decoder'):
    cond_dist = config['cond_dist']
    if cond_dist == 'gauss':
        num_out = 2
    else:
        num_out = 1
    model_func = decoder_dict[model_name]
    return tf.make_template(
        scope, model_func, config=config, num_out=num_out
    )

def get_decoder_mean(decoder_out, config):
    return tf.sigmoid(decoder_out[0])

def get_reconstr_err(decoder_out, x, config):
    cond_dist = config['cond_dist']
    if cond_dist == 'gauss':
        loc = tf.sigmoid(decoder_out[0])
        logscale = decoder_out[1]
        err = (x - loc)*tf.exp(-logscale)
        reconst_err = tf.reduce_sum(
            0.5 * err * err + logscale + 0.5 * np.log(2*np.pi),
            [1, 2, 3]
        )
    elif cond_dist == 'bernouille':
        p_logits = decoder_out[0]
        reconst_err = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=x),
          [1, 2, 3]
        )

    return reconst_err

def get_interpolations(decoder, z1, z2, N, config):
    z_dim = config['z_dim']
    alpha = tf.reshape(tf.linspace(0., 1., N), [1, N, 1])

    z1 =  tf.reshape(z1, [-1, 1, z_dim])
    z2 =  tf.reshape(z2, [-1, 1, z_dim])

    z_interp = alpha * z1 + (1 - alpha) * z2
    z_interp = tf.reshape(z_interp, [-1, z_dim])

    return decoder(z_interp)
