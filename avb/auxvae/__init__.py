import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_decoder_samples
from avb.utils import *
from avb.ops import *
import numpy as np

class AuxVAE(object):
    def __init__(self, encoder, decoder, encoder_aux, decoder_aux, x_real,
            z_eps, x_eps, a_eps1, a_eps2, config, beta=1, is_training=True):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_aux = encoder_aux
        self.decoder_aux = decoder_aux
        self.config = config
        self.x_real = x_real
        self.z_sampled = z_eps
        self.beta = beta

        cond_dist = config['cond_dist']
        output_size = config['output_size']
        c_dim = config['c_dim']
        z_dist = config['z_dist']

        factor = 1./(output_size * output_size * c_dim)

        # Set up adversary and contrasting distribution
        self.a_mean1, self.log_a_std1 = encoder_aux(x_real, is_training=is_training)
        self.a_std1 = tf.exp(self.log_a_std1)
        self.a1 = self.a_mean1 + a_eps1 * self.a_std1

        self.z_mean, self.log_z_std = encoder(x_real, self.a1, is_training=is_training)
        self.z_std = tf.exp(self.log_z_std)
        self.z_real = self.z_mean + z_eps * self.z_std

        self.decoder_out = decoder(self.z_real, is_training=is_training)
        self.x_reconstr_sample = get_decoder_samples(self.decoder_out, config)

        self.a_mean2, self.log_a_std2 = decoder_aux(self.x_reconstr_sample, self.z_real, is_training=is_training)
        self.a_std2 = tf.exp(self.log_a_std2)
        self.a2 = self.a_mean2 + a_eps2 * self.a_std2

        # Primal loss
        self.reconst_err = get_reconstr_err(self.decoder_out, self.x_real, config=config)
        self.KL_z = get_KL(self.z_mean, self.log_z_std)
        self.a_logq = get_pdf_gauss(self.a_mean1, self.log_a_std1, self.a1)
        self.z_logq = get_pdf_gauss(self.z_mean, self.log_z_std, self.z_real)
        self.a_logp = get_pdf_gauss(self.a_mean2, self.log_a_std2, self.a1)
        self.KL_a = -self.a_logp + self.a_logq
        self.KL = self.KL_z + self.KL_a

        self.ELBO = -self.reconst_err - self.KL
        self.loss = factor * tf.reduce_mean(self.reconst_err + beta*self.KL)

        # Mean values
        self.ELBO_mean = tf.reduce_mean(self.ELBO)
        self.KL_mean = tf.reduce_mean(self.KL)
        self.reconst_err_mean = tf.reduce_mean(self.reconst_err)


def get_KL(z_mean, log_z_std):
    z_std = tf.exp(log_z_std)
    KL = 0.5*tf.reduce_sum(-1 - 2*log_z_std + tf.square(z_std) + tf.square(z_mean), [1])
    return KL
