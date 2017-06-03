import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.validate.ais import AIS
from tqdm import tqdm
import pickle
import time
import ipdb

def run_tests(decoder, stats_scalar, stats_dist, x_test, params_posterior, energy0, get_z0, config,
        latent_dim=None, eps_scale=None):
    log_dir = config['log_dir']
    eval_dir = config['eval_dir']
    results_dir = os.path.join(eval_dir, "results")
    if latent_dim is None:
        latent_dim = config['z_dim']
    batch_size = config['batch_size']
    ais_nchains = config['test_ais_nchains']
    test_nais = config['test_nais']

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    saver = tf.train.Saver()
    ais = AIS(x_test, params_posterior,
        decoder=decoder, energy0=energy0, get_z0=get_z0, config=config, latent_dim=latent_dim, eps_scale=eps_scale)

    # Session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Load model
    sess.run(tf.global_variables_initializer())

    if load_session(sess, saver, config):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        return

    # KL and test likelihood
    print("Computing statistics...")
    stats_scalar_keys = stats_scalar.keys()
    stats_dist_keys = stats_dist.keys()

    stats = defaultdict(list)

    start_time = time.time()
    nbreak = 200

    ais_samples = np.zeros([ais_nchains, batch_size, latent_dim])

    progress_batch = tqdm(range(nbreak), desc="Test")
    for i in progress_batch:
        if coord.should_stop():
            break
        stats_scalar_new, stats_dist_new = get_statistics(sess, stats_scalar, stats_dist)
        for k in stats_scalar_new.keys():
            stats[k].append(stats_scalar_new[k])
        for k in stats_dist_new.keys():
            stats[k + "_mean"].append(np.mean(stats_dist_new[k]))
            stats[k + "_std"].append(np.std(stats_dist_new[k]))

    # KL
    progress_batch = tqdm(range(test_nais), desc="Test (AIS)")
    for i in progress_batch:
        if coord.should_stop():
            break

        process_stats(stats,
            save_txt=os.path.join(results_dir, "results_%d.txt" % i),
            save_pickle=os.path.join(results_dir, "results_%d.pickle" % i)
        )

        ais_res = np.zeros([ais_nchains, batch_size])
        ais.read_batch(sess)

        progress_ais = tqdm(range(ais_nchains), desc="AIS")
        for j in progress_ais:
            ais_res[j], ais_samples[j] = ais.evaluate(sess)
            ais_lprob, ais_ess = ais.average_weights(ais_res[:j+1], axis=0)
            progress_ais.set_postfix(
                lprob="%.2f+-%.2f" % (ais_lprob.mean(), ais_lprob.std()),
                ess="%.2f/%d" % (ais_ess.mean(), j+1)
            )

        stats["ais_mean"].append(np.mean(ais_lprob))
        stats["ais_std"].append(np.std(ais_lprob))
        stats["ais_ess_mean"].append(np.mean(ais_ess))
        stats["ais_ess_std"].append(np.std(ais_ess))


    # End Session
    coord.request_stop()
    coord.join(threads)
    sess.close()

    ais_samples = ais_samples.reshape(-1, z_dim)

    # Write statistics string
    statistics_str = process_stats(stats,
        save_txt=os.path.join(eval_dir, "results.txt"),
        save_pickle=os.path.join(eval_dir, "results.pickle")
    )

    print("\n" + statistics_str +"\n")

def get_statistics(sess, stats_scalar, stats_dist):
    stats_scalar_vals = sess.run(list(stats_scalar.values()))
    stats_dist_vals = sess.run(list(stats_dist.values()))

    stats_scalar = dict(zip(stats_scalar.keys(), stats_scalar_vals))
    stats_dist = dict(zip(stats_dist.keys(), stats_dist_vals))

    return stats_scalar, stats_dist

def process_stats(stats, save_txt=None, save_pickle=None):
    # Write statistics string

    statistics_str = "\n".join([
        "Statistics",
        "==========\n",
    ])

    for k in sorted(stats.keys()):
        v_list = stats[k]
        v_mean = np.mean(v_list)
        v_std = np.std(v_list)/np.sqrt(np.size(v_list))
        statistics_str += "{} = {:0.5f} +- {:0.5f}\n".format(k, v_mean, v_std)

    # Save if required
    if save_pickle is not None:
        with open(save_pickle, "wb") as f:
            pickle.dump(stats, f)

    if save_txt is not None:
        with open(save_txt, "w") as f:
            f.write(statistics_str)

    return statistics_str


def load_session(sess, saver, config):
    log_dir = config['log_dir']

    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(log_dir, ckpt_name))
        return True
    else:
        return False
