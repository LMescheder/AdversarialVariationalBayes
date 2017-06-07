import os
import scipy.misc
import numpy as np
import argparse

from avb.utils import pp
from avb import inputs
from avb.avb.train import train
from avb.avb.test import test
from avb.decoders import get_decoder
from avb.avb.models import get_encoder, get_adversary
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train and run a avae.')
parser.add_argument("--nsteps", default=200000, type=int, help="Iterations to train.")
parser.add_argument("--learning-rate", default=1e-4, type=float, help="Learning rate of for adam.")
parser.add_argument("--learning-rate-adversary", default=1e-4, type=float, help="Learning rate of for adam.")
parser.add_argument("--ntest", default=100, type=int, help="How often to run test code.")

parser.add_argument("--batch-size", default=64, type=int, help="The size of batch images.")
parser.add_argument("--image-size", default=108, type=int, help="The size of image to use (will be center cropped).")
parser.add_argument("--output-size", default=64, type=int, help="The size of the output images to produce.")

parser.add_argument("--encoder", default="conv0", type=str, help="Architecture to use.")
parser.add_argument("--decoder", default="conv0", type=str, help="Architecture to use.")
parser.add_argument("--adversary", default="conv0", type=str, help="Architecture to use.")

parser.add_argument("--c-dim", default=3, type=int, help="Dimension of image color. ")
parser.add_argument("--z-dim", default=100, type=int, help="Dimension of latent space.")
parser.add_argument("--z-dist", default="gauss", type=str, help="Prior distribution of latent space.")
parser.add_argument("--cond-dist", default="gauss", type=str, help="Conditional distribution.")
parser.add_argument("--eps-dim", default=0, type=int, help="Dimension of noise for encoder per pixel. ")
parser.add_argument("--eps-nbasis", default=32, type=int, help="Number of noise basis vectors (if needed).")
parser.add_argument("--anneal-steps", default="0", type=int, help="How many steps to use for annealing.")
parser.add_argument("--is-anneal", default=False, action='store_true', help="True for training, False for testing.")

parser.add_argument("--dataset", default="celebA", type=str, help="The name of dataset.")
parser.add_argument("--data-dir", default="data", type=str, help="Path to the data directory.")
parser.add_argument('--split-dir', default='data/splits', type=str, help='Folder where splits are found')

parser.add_argument("--log-dir", default="tf_logs", type=str, help="Directory name to save the checkpoints.")
parser.add_argument("--sample-dir", default="samples", type=str, help="Directory name to save the image samples.")
parser.add_argument("--eval-dir", default="eval", type=str, help="Directory where to save logs.")

parser.add_argument("--is-train", default=False, action='store_true', help="True for training, False for testing.")
parser.add_argument("--is-01-range", default=False,  action='store_true', help="If image is constrained to values between 0 and 1.")
parser.add_argument("--is-ac", default=False, action='store_true', help="Wether to use local normalization (only supported by some models).")

parser.add_argument("--test-nite", default=0, type=int, help="Number of iterations of ite.")
parser.add_argument("--test-nais", default=10, type=int, help="Number of iterations of ais.")
parser.add_argument("--test-ais-nchains", default=16, type=int, help="Number of chains for ais.")
parser.add_argument("--test-ais-nsteps", default=100, type=int, help="Number of annealing steps for ais.")
parser.add_argument("--test-ais-eps", default=1e-2, type=float, help="Stepsize for AIS.")
parser.add_argument("--test-is-center-posterior", default=False, action='store_true', help="Wether to center posterior plots.")


def main():
    args = parser.parse_args()
    config = vars(args)
    config['gf_dim'] = 64
    config['df_dim'] = 64
    config['test_is_adaptive_eps'] = False
    pp.pprint(config)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    decoder = get_decoder(args.decoder, config)
    encoder = get_encoder(args.encoder, config)
    adversary = get_adversary(args.adversary, config)

    if args.is_train:
        x_train = inputs.get_inputs('train', config)
        x_val = inputs.get_inputs('val', config)

        train(encoder, decoder, adversary, x_train, x_val, config)
    else:
        x_test = inputs.get_inputs('test', config)
        test(encoder, decoder, adversary, x_test, config)

if __name__ == '__main__':
    main()
