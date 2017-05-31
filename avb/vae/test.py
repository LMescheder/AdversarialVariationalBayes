import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.ops import *
from avb.validate import run_tests
from avb.validate.ais import AIS
from avb.vae import VAE
from tqdm import tqdm
import time

def test(encoder, decoder, x_test, config):
    log_dir = config['log_dir']
    eval_dir = config['eval_dir']
    results_dir = os.path.join(eval_dir, "results")
    z_dim = config['z_dim']
    batch_size = config['batch_size']
    ais_nchains = config['test_ais_nchains']
    test_nais = config['test_nais']

    z_sampled = tf.random_normal([batch_size, z_dim])
    vae_test = VAE(encoder, decoder, x_test, z_sampled, config, is_training=False)

    stats_scalar = {
        'loss': vae_test.loss,
    }

    stats_dist = {
        'ELBO': vae_test.ELBO,
        'KL': vae_test.KL,
        'reconst_err': vae_test.reconst_err,
        'z': vae_test.z_real,
    }

    params_posterior = [vae_test.z_mean, vae_test.log_z_std]
    eps_scale = vae_test.z_std

    def energy0(z, theta):
        z_mean = theta[0]
        log_z_std = theta[1]
        return -get_pdf_gauss(z_mean, log_z_std, z)

    run_tests(decoder, stats_scalar, stats_dist,
        vae_test.x_real, vae_test.z_real, params_posterior, energy0, config,
        eps_scale=eps_scale
    )
