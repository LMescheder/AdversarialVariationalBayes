import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.validate import run_tests
from avb.validate.ais import AIS
from avb.avb import AVB
from tqdm import tqdm
import time

def test(encoder, decoder, adversary, x_test, config):
    log_dir = config['log_dir']
    eval_dir = config['eval_dir']
    results_dir = os.path.join(eval_dir, "results")
    z_dim = config['z_dim']
    batch_size = config['batch_size']
    ais_nchains = config['test_ais_nchains']
    test_nais = config['test_nais']
    is_ac = config['is_ac']

    z_sampled = tf.random_normal([batch_size, z_dim])
    avb_test = AVB(encoder, decoder, adversary, x_test, z_sampled, config, is_training=False)

    stats_scalar = {
        'loss_primal': avb_test.loss_primal,
        'loss_dual': avb_test.loss_dual,
    }

    stats_dist = {
        'ELBO': avb_test.ELBO,
        'KL': avb_test.KL,
        'reconst_err': avb_test.reconst_err,
        'z': avb_test.z_real,
    }

    run_tests(decoder, stats_scalar, stats_dist,
        avb_test.x_real, avb_test.z_mean, avb_test.z_std, config
    )
