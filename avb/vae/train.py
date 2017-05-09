import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.vae import VAE
from tqdm import tqdm

def train(encoder, decoder, x_train, x_val, config):
    batch_size = config['batch_size']
    z_dim = config['z_dim']
    z_dist = config['z_dist']
    anneal_steps = config['anneal_steps']
    is_anneal = config['is_anneal']

    # TODO: support uniform
    if config['z_dist'] != 'gauss':
        raise NotImplementedError

    z_sampled = tf.random_normal([batch_size, z_dim])

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Build graphs
    if is_anneal:
        beta = tf.train.polynomial_decay(0., global_step, anneal_steps, 1.0, power=1)
    else:
        beta = 1
    vae_train = VAE(encoder, decoder, x_train, z_sampled, config, beta=beta)
    vae_val = VAE(encoder, decoder, x_val, z_sampled, config, is_training=False)

    x_fake = get_decoder_mean(decoder(z_sampled, is_training=False), config)

    # Interpolations
    z1 = tf.random_normal([8, z_dim])
    z2 = tf.random_normal([8, z_dim])
    x_interp = get_decoder_mean(get_interpolations(decoder, z1, z2, 8, config), config)

    # Variables
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    # Train op
    train_op = get_train_op(
        vae_train.loss, encoder_vars + decoder_vars, config
    )


    # Summaries
    summary_op = tf.summary.merge([
        tf.summary.scalar('train/loss', vae_train.loss),
        tf.summary.scalar('train/ELBO', vae_train.ELBO_mean),
        tf.summary.scalar('train/KL', vae_train.KL_mean),
        tf.summary.scalar('train/reconstr_err', vae_train.reconst_err_mean),
        tf.summary.scalar('val/ELBO', vae_val.ELBO_mean),
        tf.summary.scalar('val/KL', vae_val.KL_mean),
        tf.summary.scalar('val/reconstr_err', vae_val.reconst_err_mean),
    ])

    # Supervisor
    sv = tf.train.Supervisor(
        logdir=config['log_dir'], global_step=global_step,
        summary_op=summary_op, save_summaries_secs=15,
    )

    # Test samples
    z_test_np = np.random.randn(batch_size, z_dim)
    # Session
    with sv.managed_session() as sess:
        # Show real data
        samples = sess.run(x_val)
        save_images(samples[:64], [8, 8], config['sample_dir'], 'real.png')
        save_images(samples[:8], [8, 1], config['sample_dir'], 'real2.png')
        save_images(samples[8:16], [8, 1], config['sample_dir'], 'real1.png')

        # For interpolations
        z_out = sess.run(vae_val.z_real, feed_dict={x_val: samples})

        progress = tqdm(range(config['nsteps']))
        for batch_idx in progress:
            if sv.should_stop():
               break

            niter = sess.run(global_step)

            # Train
            sess.run(train_op)

            ELBO_out, ELBO_val_out = sess.run([vae_train.ELBO_mean, vae_val.ELBO_mean])

            progress.set_description('ELBO: %4.4f, ELBO (val): %4.4f'
                % (ELBO_out, ELBO_val_out))

            sess.run(global_step_op)
            if np.mod(niter, config['ntest']) == 0:
                # Test
                samples = sess.run(x_fake, feed_dict={z_sampled: z_test_np})
                save_images(samples[:64], [8, 8], os.path.join(config['sample_dir'], 'samples'),
                            'train_{:06d}.png'.format(niter)
                )

                interplations = sess.run(x_interp, feed_dict={z1: z_out[:8], z2: z_out[8:16]})
                save_images(interplations[:64], [8, 8], os.path.join(config['sample_dir'], 'interp'),
                            'interp_{:06d}.png'.format(niter)
                )


def get_train_op(loss, vars, config):
    learning_rate = config['learning_rate']

    # Train step
    optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True, beta1=0.5)
    train_op = optimizer.minimize(loss, var_list=vars)

    return train_op
