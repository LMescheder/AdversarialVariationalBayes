import tensorflow as tf
import numpy as np
import scipy as sp
import time
from tqdm import tqdm

class AIS(object):
    def __init__(self, sess, decoder, error, z_dim, batch_size,
            eps, output_size, c_dim, L=30, nsteps=10000, adaptive_eps=False, sigma=1.):
        self.sess = sess
        self.decoder = decoder
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.error = error
        self.sigma = sigma

        self.eps0 = eps
        self.output_size = output_size
        self.c_dim = c_dim

        self.L = L
        self.nsteps = nsteps
        self.adaptive_eps = adaptive_eps

        self.build_model()

    def build_model(self):
        # Input
        self.x_in = tf.placeholder(tf.float32)
        self.mean0_in = tf.placeholder(tf.float32)
        self.std0_in = tf.placeholder(tf.float32)

        # Persist on gpu for efficiency
        self.x = tf.Variable(np.zeros([self.batch_size, self.output_size, self.output_size, self.c_dim], dtype=np.float32), trainable=False)
        self.mean0 = tf.Variable(np.zeros([self.batch_size, self.z_dim], dtype=np.float32), trainable=False)
        self.std0 = tf.Variable(np.ones([self.batch_size, self.z_dim], dtype=np.float32), trainable=False)
        self.var0 = tf.square(self.std0)

        # Position and momentum variables
        mass = 1.#/self.var0
        mass_sqrt = 1.#/self.std0
        self.z = tf.Variable(np.zeros([self.batch_size, self.z_dim], dtype=np.float32), trainable=False)
        self.p = tf.Variable(np.zeros([self.batch_size, self.z_dim], dtype=np.float32), trainable=False)

        self.z_current = tf.Variable(np.zeros([self.batch_size, self.z_dim], dtype=np.float32), trainable=False)
        self.p_current = tf.Variable(np.zeros([self.batch_size, self.z_dim], dtype=np.float32), trainable=False)

        self.p_rnd = tf.random_normal([self.batch_size, self.z_dim]) * mass_sqrt

        self.eps = tf.placeholder(tf.float32, shape=[])
        self.beta = tf.placeholder(tf.float32, shape=[])

        # Hamiltoninan

        self.U = self.get_energy(self.z)
        self.V = 0.5 * tf.reduce_sum(tf.square(self.p)/mass, [1])
        self.H = self.U + self.V
        self.U_current = self.get_energy(self.z_current)
        self.V_current = 0.5 * tf.reduce_sum(tf.square(self.p_current)/mass, [1])
        self.H_current = self.U_current + self.V_current

        # Intialize
        self.init_hmc = [
            self.x.assign(self.x_in),
            self.mean0.assign(self.mean0_in),
            self.std0.assign(self.std0_in)
        ]

        self.init_z = self.z_current.assign(
            self.mean0 + self.std0 * tf.random_normal([self.batch_size, self.z_dim])
        )

        self.init_hmc_step = [
            # self.z.assign(self.z_current),
            # self.p.assign(self.p_rnd),
            self.p_current.assign(self.p_rnd)
        ]

        self.init_hmc_step2 = [
            self.z.assign(self.z_current),
            self.p.assign(self.p_current),
        ]
        # Euler steps
        eps_scaled = self.std0 * self.eps

        self.euler_z = self.z.assign_add(eps_scaled * self.p/mass)
        gradU = tf.reshape(tf.gradients(self.U, self.z), [self.batch_size, self.z_dim])
        self.euler_p = self.p.assign_sub(eps_scaled * gradU)

        # Accept
        self.is_accept = tf.cast(tf.random_uniform([self.batch_size]) < tf.exp(self.H_current - self.H), tf.float32)
        self.accept_rate = tf.reduce_mean(self.is_accept)

        is_accept_rs = tf.reshape(self.is_accept, [self.batch_size, 1])
        self.update_z = self.z_current.assign(
            is_accept_rs * self.z + (1. - is_accept_rs) * self.z_current
        )

    def get_energy(self, z):
        E = self.beta*self.get_energy1(z) + (1 - self.beta) * self.get_energy0(z)
        return E

    def get_energy1(self, z):
        self.x_logits_, self.xstd_logits_ = self.decoder.logits(z, is_training=False)
        self.x_, self.xstd_ = self.decoder(z, is_training=False)

        if self.error == "cross_entropy":
            E = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits_, labels=self.x), [1, 2, 3]
            )
        elif self.error == "gauss":
            err = (self.x_ - self.x)/self.xstd_
            E = tf.reduce_sum(
                 0.5 * err * err + tf.log(self.xstd_) + 0.5 * np.log(2*np.pi), [1, 2, 3]
            )
        elif self.error == "gauss_fixed":
            err = (self.x_ - self.x)/self.sigma
            E = tf.reduce_sum(
                 0.5 * err * err + np.log(self.sigma) + 0.5 * np.log(2*np.pi), [1, 2, 3]
            )
        # elif self.error == "laplace":
        #     err = (self.x_ - self.x)/self.xstd_
        #     E = (
        #         tf.reduce_sum(tf.abs(err))
        #         + tf.reduce_sum(tf.log(self.xstd_) + np.log(2))
        #     )
        else:
            raise ValueError("Invalid value for parameter `error`.")

        # Prior
        E += tf.reduce_sum(
            0.5 * tf.square(z) + 0.5 * np.log(2*np.pi), [1]
        )

        return E

    def get_energy0(self, z):
        z_norm = (z - self.mean0) / self.std0
        E = tf.reduce_sum(
            0.5 * z_norm * z_norm + tf.log(self.std0) + 0.5 * np.log(2*np.pi), [1]
        )
        return E

    def run_hmc_step(self, beta, eps):
        # Initialize
        self.sess.run(self.init_hmc_step)
        self.sess.run(self.init_hmc_step2)

        # Leapfrog steps
        self.sess.run(self.euler_p, feed_dict={self.eps: eps/2, self.beta: beta})
        for i in range(self.L+1):
            self.sess.run(self.euler_z, feed_dict={self.eps: eps, self.beta: beta})
            if i < self.L:
                self.sess.run(self.euler_p, feed_dict={self.eps: eps, self.beta: beta})
        self.sess.run(self.euler_p, feed_dict={self.eps: eps/2, self.beta: beta})

        # Update z
        _, accept_rate = self.sess.run([self.update_z, self.accept_rate], feed_dict={self.beta: beta})
        return accept_rate

    def evaluate(self, x_test, mean0=None, std0=None):
        if mean0 is None:
            mean0 = np.zeros([self.batch_size, self.z_dim], dtype=np.float32)
        if std0 is None:
            std0 = np.ones([self.batch_size, self.z_dim], dtype=np.float32)

        # logZ = self.sess.run(tf.reduce_sum(tf.log(self.std0), [1]))
        # logZ = self.z_dim * 0.5 * np.log(2*np.pi)
        logpx = 0.
        weights = np.zeros([100, self.batch_size])

        betas = np.linspace(0, 1, self.nsteps+1)
        accept_rate = 1.
        eps = self.eps0

        feed_dict = {
            self.x_in: x_test,
            self.mean0_in: mean0, self.std0_in: std0
        }

        self.sess.run(self.init_hmc, feed_dict=feed_dict)
        self.sess.run(self.init_z)

        t = time.time()
        progress = tqdm(range(self.nsteps), desc="HMC")
        for i in progress:
            f0 = -self.sess.run(self.U_current, feed_dict={self.beta: betas[i]})
            f1 = -self.sess.run(self.U_current, feed_dict={self.beta: betas[i+1]})
            logpx += f1 - f0

            if i < self.nsteps-1:
                accept_rate = self.run_hmc_step(betas[i+1], eps)
                if self.adaptive_eps and accept_rate < 0.6:
                    eps = eps / 1.1
                elif self.adaptive_eps and accept_rate > 0.7:
                    eps = eps * 1.1
                progress.set_postfix(
                    accept_rate="%.2f" % accept_rate,
                    logw="%.2f+-%.2f" % (logpx.mean(), logpx.std()),
                    eps="%.2e" % eps,
                )
        samples = self.sess.run(self.z_current)

        return logpx, samples

    def average_weights(self, weights, axis=0):
        nchains = weights.shape[axis]
        logsumw = sp.misc.logsumexp(weights, axis=axis)
        lprob = logsumw - np.log(nchains)
        ess = np.exp(-sp.misc.logsumexp(2*(weights - logsumw), axis=axis))
        return lprob, ess
