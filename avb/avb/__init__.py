import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *

class AVB(object):
    def __init__(self, encoder, decoder, adversary, x_real, z_sampled, config, is_training=True):
        self.encoder = encoder
        self.decoder = decoder
        self.adversary = adversary
        self.config = config
        self.x_real = x_real
        self.z_sampled = z_sampled

        is_ac = config['is_ac']
        cond_dist = config['cond_dist']
        output_size = config['output_size']
        c_dim = config['c_dim']
        z_dist = config['z_dist']

        factor = 1./(output_size * output_size * c_dim)

        # Set up adversary and contrasting distribution
        if not is_ac:
            self.z_real = encoder(x_real, is_training=is_training)
            self.z_mean, self.z_var = tf.zeros_like(self.z_real), tf.ones_like(self.z_real)
            self.z_std = tf.ones_like(self.z_real)
        else:
            self.z_real, z_mean, z_var = encoder(x_real, is_training=is_training)
            self.z_mean, self.z_var = tf.stop_gradient(z_mean), tf.stop_gradient(z_var)
            self.z_std = tf.sqrt(z_var + 1e-4)

        self.z_norm = (self.z_real - self.z_mean)/self.z_std
        Td = adversary(self.z_norm, x_real, is_training=is_training)
        Ti = adversary(self.z_sampled, x_real, is_training=is_training)
        logz = get_zlogprob(self.z_real, z_dist)
        logr = -0.5 * tf.reduce_sum(self.z_norm*self.z_norm + tf.log(self.z_var) + np.log(2*np.pi), [1])

        self.decoder_out = decoder(self.z_real, is_training=is_training)

        # Primal loss
        self.reconst_err = get_reconstr_err(self.decoder_out, self.x_real, config=config)
        self.KL = Td + logr - logz
        self.ELBO = self.reconst_err + self.KL
        self.loss_primal = factor * tf.reduce_mean(self.ELBO)

        # Mean values
        self.ELBO_mean = tf.reduce_mean(self.ELBO)
        self.KL_mean = tf.reduce_mean(self.KL)
        self.reconst_err_mean = tf.reduce_mean(self.reconst_err)

        # Dual loss
        d_loss_d = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td))
        )
        d_loss_i = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti))
        )

        self.loss_dual = d_loss_i + d_loss_d

def get_zlogprob(z, z_dist):
    if z_dist == "gauss":
        logprob = -0.5 * tf.reduce_sum(z*z  + np.log(2*np.pi), [1])
    elif z_dist == "uniform":
        logprob = 0.
    else:
        raise ValueError("Invalid parameter value for `z_dist`.")
    return logprob
