import os
import scipy.misc
import numpy as np

from autoencoders.vae import VAE
from autoencoders.loader import (
    ImageLoader, MNISTLoader, CIFARLoader, FakeLoader
)
from autoencoders.utils import pp
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("nepochs", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("momentum", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("clip_gradient", 1., "Gradient clipping [1.]")
flags.DEFINE_float("ntest", 100, "How often to run test code.")
flags.DEFINE_float("npretrain", 0, "Number of steps for pretraining.")
flags.DEFINE_integer("beta_niter", 0, "Number of iterations to increase beta from 0 to 1 [0].")

flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")

flags.DEFINE_string("architecture", "conv0", "Architecture to use [conv0, full0].")
flags.DEFINE_string("error", "gauss", "Architecture to use [gauss, gauss_fixed, cross_entropy].")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color [3]. ")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent space [3].")
flags.DEFINE_string("z_dist", "uniform", "Prior distribution of latent space.")
flags.DEFINE_integer("eps_dim", 0, "Dimension of noise for encoder per pixel [5]. ")
flags.DEFINE_integer("eps_c_dim", 5, "Dimension of noise for encoder per pixel [0]. ")
flags.DEFINE_float("sigma", 0.01, "Standard deviation of image noise [0.01]. ")

flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("data_dir", "data", "Path to the data directory [data].")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint].")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples].")
flags.DEFINE_string("log_dir", "logs", "Directory where to save logs [logs].")
flags.DEFINE_string("eval_dir", "eval", "Directory where to save logs [eval].")

flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "Crop input images [True].")
flags.DEFINE_boolean("is_clip_gradients", False, "True for training, False for testing [False].")
flags.DEFINE_boolean("is_01_range", True, "If image is constrained to values between 0 and 1 [True].")
flags.DEFINE_boolean("is_iwae", False, "If importance weighting should be used (per batch).")
flags.DEFINE_integer("n_iwae", 16, "Number of importance samples. batch_size must be multiple of this.")

flags.DEFINE_integer("test_nite", 0, "Number of iterations of ite.")
flags.DEFINE_integer("test_nais", 1, "Number of iterations of ais.")
flags.DEFINE_integer("test_ais_nchains", 16, "Number of chains for ais.")
flags.DEFINE_integer("test_ais_nsteps", 100, "Number of annealing steps for ais.")
flags.DEFINE_float("test_ais_eps", 1e-2, "Stepsize for AIS.")
flags.DEFINE_integer("test_is_center_posterior", False, "Wether to center posterior plots.")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    if FLAGS.dataset == 'mnist':
        data_loader = MNISTLoader(
            os.path.join(FLAGS.data_dir, FLAGS.dataset),
            batch_size=FLAGS.batch_size
        )
        FLAGS.output_size = 28
        FLAGS.c_dim = 1
    elif FLAGS.dataset == "cifar-10":
        data_loader = CIFARLoader(
            os.path.join(FLAGS.data_dir, FLAGS.dataset),
            batch_size=FLAGS.batch_size
        )
        FLAGS.output_size = 32
        FLAGS.c_dim = 3
    elif FLAGS.dataset == "fake":
        FLAGS.output_size = 2
        FLAGS.c_dim = 1
        data_loader = FakeLoader(
            batch_size=FLAGS.batch_size,
            output_size=FLAGS.output_size,
            c_dim=FLAGS.c_dim,
            nstates=4
        )
    else:
        data_loader = ImageLoader(
            os.path.join(FLAGS.data_dir, FLAGS.dataset),
            batch_size=FLAGS.batch_size, image_size=FLAGS.image_size, output_size=FLAGS.output_size,
            c_dim=FLAGS.c_dim, is_crop=FLAGS.is_crop
        )


    with tf.Session() as sess:
        vae = VAE(
            sess=sess, batch_size=FLAGS.batch_size, sample_size=64, output_size=FLAGS.output_size,
            architecture=FLAGS.architecture, error=FLAGS.error,
            c_dim=FLAGS.c_dim, z_dim=FLAGS.z_dim, z_dist=FLAGS.z_dist,
            sigma=FLAGS.sigma,
            dataset_name=FLAGS.dataset,
            beta_niter=FLAGS.beta_niter,
            is_01_range=FLAGS.is_01_range,
            is_iwae=FLAGS.is_iwae,
            test_ais_nsteps=FLAGS.test_ais_nsteps,
            test_ais_eps=FLAGS.test_ais_eps,
        )

        if FLAGS.is_train:
            vae.train(data_loader, FLAGS)
        else:
            vae.test(data_loader, FLAGS)


if __name__ == '__main__':
    tf.app.run()
