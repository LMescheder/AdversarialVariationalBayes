import os
from subprocess import call
from os import path

# Executables
executable = 'python'

# Paths
srcdir = '../..'
scriptname = 'run_avae.py'
cwd = os.path.dirname(os.path.abspath(__file__))
outdir = cwd
rootdir = srcdir

# Arguments
args = [
# Architecture
 '--is-train',
'--image-size', '375',
'--output-size', '28',
'--c-dim', '1',
'--z-dim', '32',
'--z-dist', 'gauss',
'--cond-dist', 'bernouille',
'--eps-dim', '64',
'--eps-nbasis', '32',
'--encoder', 'conv0_ac',
'--decoder', 'conv0',
'--adversary', 'conv0',
'--is-01-range',
'--is-ac',
# Training
'--nsteps', '2500000',
'--ntest', '100',
"--learning-rate", "5e-5",
"--learning-rate-adversary", "1e-4",
'--batch-size', '128',
'--log-dir', os.path.join(outdir, 'logs'),
'--sample-dir', os.path.join(outdir, 'samples'),
# Data set
'--dataset', 'mnist',
'--data-dir', 'datasets',
'--split-dir', 'datasets/splits',
# Test
'--eval-dir', os.path.join(outdir, 'eval'),
'--test-nite', '0',
'--test-nais', '20',
'--test-ais-nsteps', '2000', 
'--test-ais-nchains', '8',
'--test-ais-eps', '1e-2',
'--test-is-center-posterior',
]


# Set environment variables here
my_env = os.environ.copy()
# Run
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
