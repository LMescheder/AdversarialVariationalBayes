import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *

class VAE(object):
    def __init__(self, encoder, decoder, x_real, z_sampled, config, is_training=True):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.x_real = x_real
        self.z_sampled = z_sampled

        cond_dist = config['cond_dist']
        output_size = config['output_size']
        c_dim = config['c_dim']
        z_dist = config['z_dist']

        factor = 1./(output_size * output_size * c_dim)

        # Set up adversary and contrasting distribution
        self.mean_z, self.log_std_z = encoder(x_real)
        self.std_z = tf.exp(self.log_std_z)
        self.z_real = self.mean_z + z_sampled * self.std_z
        self.decoder_out = decoder(z_sampled)

        # Primal loss
        self.reconst_err = get_reconstr_err(self.decoder_out, self.x_real, config=config)
        self.KL = get_KL(self.mean_z, self.log_std_z, z_dist)
        self.ELBO = -self.reconst_err - self.KL
        self.loss = -factor * tf.reduce_mean(self.ELBO)

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
