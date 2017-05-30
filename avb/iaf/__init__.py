import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *

class IAFVAE(object):
    def __init__(self, encoder, decoder, iaf_layers, x_real, z_sampled, config, beta=1, is_training=True):
        self.encoder = encoder
        self.decoder = decoder
        self.iaf_layers = iaf_layers
        self.config = config
        self.x_real = x_real
        self.z_sampled = z_sampled
        self.beta = beta

        cond_dist = config['cond_dist']
        output_size = config['output_size']
        c_dim = config['c_dim']
        z_dim = config['z_dim']
        z_dist = config['z_dist']

        factor = 1./(output_size * output_size * c_dim)

        # Set up adversary and contrasting distribution
        self.z_mean, self.log_z_std, self.a = encoder(x_real, is_training=is_training)
        self.z_std = tf.exp(self.log_z_std)
        z = self.z_mean + z_sampled * self.z_std

        self.KL = get_KL(self.z_mean, self.log_z_std, z_dist)

        # IAF layers
        for iaf_layer in iaf_layers:
            m, s = iaf_layer(z, self.a, activation_fn=tf.nn.elu)
            sigma = tf.sigmoid(s)
            z = sigma * z + (1 - sigma) * m
            self.KL -= tf.reduce_sum(tf.log(sigma + 1e-8), [1])
            z = tf.reverse(z, axis=[1])
        self.z_real = z
        self.decoder_out = decoder(self.z_real, is_training=is_training)

        # Primal loss
        self.reconst_err = get_reconstr_err(self.decoder_out, self.x_real, config=config)
        self.ELBO = -self.reconst_err - self.KL
        self.loss = factor * tf.reduce_mean(self.reconst_err + beta*self.KL)

        # Mean values
        self.ELBO_mean = tf.reduce_mean(self.ELBO)
        self.KL_mean = tf.reduce_mean(self.KL)
        self.reconst_err_mean = tf.reduce_mean(self.reconst_err)

def get_KL(z_mean, log_z_std, z_dist):
    if z_dist == "gauss":
        z_std = tf.exp(log_z_std)
        KL = 0.5*tf.reduce_sum(-1 - 2*log_z_std + z_std*z_std + z_mean*z_mean, [1])
    else:
        raise NotImplementedError

    return KL
