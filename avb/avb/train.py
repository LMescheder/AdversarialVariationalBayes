import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from avb.avb import AVB
from tqdm import tqdm

def train(encoder, decoder, adversary, x_train, x_val, config):
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
    avb_train = AVB(encoder, decoder, adversary, x_train, z_sampled, config, beta=beta)
    avb_val = AVB(encoder, decoder, adversary, x_val, z_sampled, config, is_training=False)

    x_fake = get_decoder_mean(decoder(z_sampled, is_training=False), config)

    # Interpolations
    z1 = tf.random_normal([8, z_dim])
    z2 = tf.random_normal([8, z_dim])
    x_interp = get_decoder_mean(get_interpolations(decoder, z1, z2, 8, config), config)

    # Variables
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    adversary_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversary')

    # Train op
    train_op = get_train_op(
        avb_train.loss_primal, avb_train.loss_dual,
        encoder_vars + decoder_vars, adversary_vars, config
    )

    # Summaries
    summary_op = tf.summary.merge([
        tf.summary.scalar('train/loss_primal', avb_train.loss_primal),
        tf.summary.scalar('train/loss_dual', avb_train.loss_dual),
        tf.summary.scalar('train/ELBO', avb_train.ELBO_mean),
        tf.summary.scalar('train/KL', avb_train.KL_mean),
        tf.summary.scalar('train/reconstr_err', avb_train.reconst_err_mean),
        tf.summary.scalar('val/ELBO', avb_val.ELBO_mean),
        tf.summary.scalar('val/KL', avb_val.KL_mean),
        tf.summary.scalar('val/reconstr_err', avb_val.reconst_err_mean),
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
        z_out = sess.run(avb_val.z_real, feed_dict={x_val: samples})

        progress = tqdm(range(config['nsteps']))
        for batch_idx in progress:
            if sv.should_stop():
               break

            niter = sess.run(global_step)

            # Train
            sess.run(train_op)

            ELBO_out, ELBO_val_out = sess.run([avb_train.ELBO_mean, avb_val.ELBO_mean])

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


def get_train_op(loss_primal, loss_dual, vars_primal, vars_dual, config):
    learning_rate = config['learning_rate']
    learning_rate_adversary = config['learning_rate_adversary']

    # Train step
    primal_optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True, beta1=0.5)
    adversary_optimizer = tf.train.AdamOptimizer(learning_rate_adversary, use_locking=True, beta1=0.5)

    primal_grads = primal_optimizer.compute_gradients(loss_primal, var_list=vars_primal)
    adversary_grads = adversary_optimizer.compute_gradients(loss_dual, var_list=vars_dual)

    primal_grads = [(grad, var) for grad, var in primal_grads if grad is not None]
    adversary_grads = [(grad, var) for grad, var in adversary_grads if grad is not None]

    allgrads = [grad for grad, var in primal_grads + adversary_grads]
    with tf.control_dependencies(allgrads):
        primal_train_step = primal_optimizer.apply_gradients(primal_grads)
        adversary_train_step = adversary_optimizer.apply_gradients(adversary_grads)

    train_op = tf.group(primal_train_step, adversary_train_step)

    return train_op
