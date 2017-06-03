import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.ops import *
from avb.validate import run_tests
from avb.validate.ais import AIS
from avb.iaf import IAFVAE, apply_iaf
from tqdm import tqdm
import time
import ipdb

def test(encoder, decoder, iaf_layers, x_test, config):
    log_dir = config['log_dir']
    eval_dir = config['eval_dir']
    results_dir = os.path.join(eval_dir, "results")
    z_dim = config['z_dim']
    batch_size = config['batch_size']
    ais_nchains = config['test_ais_nchains']
    test_nais = config['test_nais']

    z_sampled = tf.random_normal([batch_size, z_dim])
    iaf_test = IAFVAE(encoder, decoder, iaf_layers, x_test, z_sampled, config, is_training=False)

    stats_scalar = {
        'loss': iaf_test.loss,
    }

    stats_dist = {
        'ELBO': iaf_test.ELBO,
        'KL': iaf_test.KL,
        'reconst_err': iaf_test.reconst_err,
        'z': iaf_test.z_real,
    }

    params_posterior = [] #iaf_test.z_mean, iaf_test.log_z_std, iaf_test.a]

    def energy0(z, theta):
        E = tf.reduce_sum(
            0.5 * tf.square(z) + 0.5 * np.log(2*np.pi), [1]
        )
        return E
        # z_mean = theta[0]
        # log_z_std = theta[1]
        # a = theta[2]
        # logq = get_pdf_gauss(z_mean, log_z_std, z)
        #
        # # IAF layers
        # _, logq = apply_iaf(iaf_layers, a, z, logq)
        #
        # return -logq

    def get_z0(theta):
        return  tf.random_normal([batch_size, z_dim])

    run_tests(decoder, stats_scalar, stats_dist,
        iaf_test.x_real, params_posterior, energy0, get_z0, config,
    )
