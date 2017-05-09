import tensorflow as tf
from avb.decoders import get_reconstr_err, get_decoder_mean, get_interpolations
from avb.utils import *

def run_training(train_op, summary_op, x_val, z_real, x_interp, z1, z2, config):
    z_dim = config['z_dim']
    batch_size = config['batch_size']

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

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
