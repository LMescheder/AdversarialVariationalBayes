import os
from subprocess import call
from os import path

# Executables
executable = '/is/ps2/lmescheder/Apps/anaconda2/envs/tensorflow3/bin/python'

# Paths
srcdir = '../..'
scriptname = 'run_avae.py'
cwd = os.path.dirname(os.path.abspath(__file__))
outdir = cwd
rootdir = srcdir

# Arguments
args = [
# Architecture
#'--is-train',
'--image-size', '375',
'--output-size', '28',
'--c-dim', '3',
'--z-dim', '32',
'--z-dist', 'gauss',
'--cond-dist', 'bernouille',
'--eps-dim', '64',
'--eps-nbasis', '32',
'--encoder', 'conv1_ac',
'--decoder', 'conv1',
'--adversary', 'conv0',
'--is-01-range',
'--is-ac',
# Training
'--nsteps', '2500000',
'--ntest', '100',
"--learning-rate", "1e-4",
"--learning-rate-adversary", "2e-4",
'--batch-size', '64',
'--log-dir', os.path.join(outdir, 'logs'),
'--sample-dir', os.path.join(outdir, 'samples'),
# Data set
'--dataset', 'mnist',
'--data-dir', 'datasets',
'--split-dir', 'datasets/splits',
# Test
'--eval-dir', os.path.join(outdir, 'eval'),
'--test-nite', '0',
'--test-nais', '10',
'--test-ais-nsteps', '20', 
'--test-ais-nchains', '2',
'--test-ais-eps', '1e-2',
'--test-is-center-posterior',
]


# Run
my_env = os.environ.copy()
# my_env["CUDA_TOOLKIT_ROOT_DIR"] = "/usr/local/cuda-7.5"
# my_env["CUDA_BIN_PATH"] = "/usr/local/cuda-7.5"
my_env["LD_LIBRARY_PATH"] = "/is/software/nvidia/cuda-8.0/lib64/:/is/software/nvidia/cudnn-5.1/lib64/"
my_env["CUDA_VISIBLE_DEVICES"] = ""

call([executable, scriptname] + args, env=my_env, cwd=rootdir)


# process = Popen(" ".join([executable, scriptname] + args), env=my_env, shell=True, cwd=rootdir)
# process.communicate()
