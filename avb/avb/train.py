import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *
from tqdm import tqdm

def train(encoder, decoder, adversary, x_train, x_val, config):
    batch_size = config['batch_size']
    z_dim = config['z_dim']
    z_dist = config['z_dist']

    # TODO: support uniform
    if config['z_dist'] != 'gauss':
        raise NotImplementedError

    z_sampled = tf.random_normal([batch_size, z_dim])

    # Losses train
    loss_primal, loss_dual, ELBO, KL, reconst_err, z_real = get_losses(encoder, decoder, adversary, x_train, z_sampled, config)
    # Losses validation
    _, _, ELBO_val, KL_val, reconst_err_val, z_real_val = get_losses(encoder, decoder, adversary, x_val, z_sampled, config, is_training=False)

    ELBO_mean, ELBO_val_mean = tf.reduce_mean(ELBO), tf.reduce_mean(ELBO_val)

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
    train_op = get_train_op(loss_primal, loss_dual, encoder_vars + decoder_vars, adversary_vars, config)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Summaries
    summary_op = tf.summary.merge([
        tf.summary.scalar('train/loss_primal', loss_primal),
        tf.summary.scalar('train/loss_dual', loss_dual),
        tf.summary.scalar('train/ELBO', tf.reduce_mean(ELBO)),
        tf.summary.scalar('train/KL', tf.reduce_mean(KL)),
        tf.summary.scalar('train/reconstr_err', tf.reduce_mean(reconst_err)),
        tf.summary.scalar('val/ELBO', tf.reduce_mean(ELBO_val)),
        tf.summary.scalar('val/KL', tf.reduce_mean(KL_val)),
        tf.summary.scalar('val/reconstr_err', tf.reduce_mean(reconst_err_val)),
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
        z_out = sess.run(z_real_val, feed_dict={x_val: samples})

        progress = tqdm(range(config['nsteps']))
        for batch_idx in progress:
            if sv.should_stop():
               break

            niter = sess.run(global_step)

            # Train
            sess.run(train_op)

            ELBO_out, ELBO_val_out = sess.run([ELBO_mean, ELBO_val_mean])

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

def get_losses(encoder, decoder, adversary, x_real, z_sampled, config, is_training=True):
    is_ac = config['is_ac']
    cond_dist = config['cond_dist']
    output_size = config['output_size']
    c_dim = config['c_dim']
    z_dist = config['z_dist']

    factor = 1./(output_size * output_size * c_dim)

    # Set up adversary and contrasting distribution
    if not is_ac:
        z_real = encoder(x_real, is_training=is_training)
        Td = adversary(z_real, x_real, is_training=is_training)
        Ti = adversary(z_sampled, x_real, is_training=is_training)
        logz = get_zlogprob(z_real, z_dist)
        logr = logz
    else:
        z_real, z_mean, z_var = encoder(x_real, is_training=is_training)
        z_mean, z_var = tf.stop_gradient(z_mean), tf.stop_gradient(z_var)
        z_std = tf.sqrt(z_var + 1e-4)
        z_norm = (z_real - z_mean)/z_std
        Td = adversary(z_norm, x_real, is_training=is_training)
        Ti = adversary(z_sampled, x_real, is_training=is_training)
        logz = get_zlogprob(z_real, z_dist)
        logr = -0.5 * tf.reduce_sum(z_norm*z_norm + tf.log(z_var) + np.log(2*np.pi), [1])

    decoder_out = decoder(z_real, is_training=is_training)

    # Primal loss
    reconst_err = get_reconstr_err(decoder_out, x_real, config=config)
    KL = Td + logr - logz
    ELBO = reconst_err + KL
    loss_primal = factor * tf.reduce_mean(ELBO)

    # Dual loss
    d_loss_d = tf.reduce_mean(
       tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td))
    )
    d_loss_i = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti))
    )

    loss_dual = d_loss_i + d_loss_d

    return loss_primal, loss_dual, ELBO, KL, reconst_err, z_real

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

def get_zlogprob(z, z_dist):
    if z_dist == "gauss":
        logprob = -0.5 * tf.reduce_sum(z*z  + np.log(2*np.pi), [1])
    elif z_dist == "uniform":
        logprob = 0.
    else:
        raise ValueError("Invalid parameter value for `z_dist`.")
    return logprob
