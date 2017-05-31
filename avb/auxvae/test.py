import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.ops import *
from avb.validate import run_tests
from avb.validate.ais import AIS
from avb.auxvae import AuxVAE
from tqdm import tqdm
import time
from tensorflow.contrib import graph_editor
import ipdb

def test(encoder, decoder, encoder_aux, decoder_aux, x_test, config):
    log_dir = config['log_dir']
    eval_dir = config['eval_dir']
    results_dir = os.path.join(eval_dir, "results")
    batch_size = config['batch_size']
    ais_nchains = config['test_ais_nchains']
    test_nais = config['test_nais']
    z_dim = config['z_dim']
    a_dim = config['a_dim']

    z_sampled = tf.random_normal([batch_size, z_dim])
    z_eps = z_sampled
    a_eps1 = tf.random_normal([batch_size, a_dim])
    a_eps2 = tf.random_normal([batch_size, a_dim])
    x_eps = tf.random_normal(tf.shape(x_test))
    auxvae_test = AuxVAE(encoder, decoder, encoder_aux, decoder_aux, x_test,
        z_eps, x_eps, a_eps1, a_eps2, config, is_training=False)

    stats_scalar = {
        'loss': auxvae_test.loss,
    }

    stats_dist = {
        'ELBO': auxvae_test.ELBO,
        'KL': auxvae_test.KL,
        'reconst_err': auxvae_test.reconst_err,
        'z': auxvae_test.z_real,
    }

    params_posterior = [x_test, auxvae_test.a_mean1, auxvae_test.log_a_std1]

    def energy0(latent, theta):
        z = latent[:, :z_dim]
        a = latent[:, z_dim:]

        x = theta[0]
        a_mean1 = theta[1]
        log_a_std1 = theta[2]

        a_logq = get_pdf_gauss(a_mean1, log_a_std1, a)
        z_mean, log_z_std = encoder(x, a, is_training=False)
        z_logq = get_pdf_gauss(z_mean, log_z_std, z)

        return - a_logq - z_logq

    latent_samples = tf.concat([auxvae_test.z_real, auxvae_test.a1], 1)

    run_tests(decoder, stats_scalar, stats_dist,
        auxvae_test.x_real, latent_samples, params_posterior, energy0, config,
        latent_dim = z_dim + a_dim,
    )
