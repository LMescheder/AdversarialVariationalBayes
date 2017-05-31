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

    params_posterior = [auxvae_test.z_mean, auxvae_test.log_z_std]

    def energy0(z, theta):
        z_mean = theta[0]
        log_z_std = theta[1]
        return -get_pdf_gauss(z_mean, log_z_std, z)


    latent_samples = auxvae_test.z_real

    run_tests(decoder, stats_scalar, stats_dist,
        auxvae_test.x_real, latent_samples, params_posterior, energy0, config,
        latent_dim = z_dim,
    )
