import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
import src.fitting as fitting
import src.multielec_utils as mutils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Dataset in format YYYY-MM-DD-P.')
parser.add_argument('-w', '--wnoise', type=str, help='White noise run in format dataXXX (or streamed/dataXXX or kilosort_dataXXX/dataXXX, etc.).')
parser.add_argument('-e', '--estim', type=str, help='Estim run in format dataXXX.')
parser.add_argument('-o', '--output', type=str, help='/path/to/output/directory.')
parser.add_argument('-t', '--threads', type=int, help='Number of threads to use in computation.')
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
parser.add_argument('-r', '--resample', type=int, help="Run resampling routine on parameters with trials")
parser.add_argument('-p', '--pattern', type=int, help="Estim pattern.")
parser.add_argument('-c', '--cell', type=int, help="Cell of interest.")

args = parser.parse_args()

### PARSE ARGUMENTS ### 

# Path definitions
ANALYSIS_BASE = "/Volumes/Analysis"
MATFILE_BASE = "/Volumes/Scratch/Users/praful/triplet_gsort_matfiles_20220420"
gsort_path = None

dataset = args.dataset
estim = args.estim
wnoise = args.wnoise
electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
vis_datapath = os.path.join(ANALYSIS_BASE, dataset, wnoise)

p = args.pattern
cell = args.cell

### DATA CLEANING ###

X_expt, probs, T = mutils.triplet_cleaning(electrical_path, gsort_path, dataset, estim, wnoise, p, cell, 
                                           load_from_mat=True, MATFILE_BASE=MATFILE_BASE)

### MULTI-SITE MODEL FITTING ###

if args.resample is not None:
    resample_weights = []
    for i in range(args.resample):
        print('Resampling {}/{}'.format(str(i+1), str(args.resample)))
        probs_resample = np.random.binomial(T, probs) / T
        resample_weights.append(fitting.fit_triplet_surface(X_expt, probs_resample, T))

    resample_weights = np.array(resample_weights)
    np.save('resampling-weights-{}-n{}-p{}-r{}.npy'.format(dataset, str(cell), str(p), str(args.resample)), 
            resample_weights)
else:
    weights = fitting.fit_triplet_surface(X_expt, probs, T)

    print(weights)
