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
parser.add_argument('-s', '--subsample', type=int, help="Subsample amplitudes")
parser.add_argument('-m', '--num_sites', type=int, help="Fixed number of sites for subsampling runs")

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

if args.subsample is not None:
    subsample_repeats = 100
    subsample_weights = []
    for i in range(subsample_repeats):
        print('Subsample Repeat {}/{}'.format(str(i+1), str(subsample_repeats)))
        random_inds = np.random.choice(len(X_expt), size=args.subsample, replace=False)
        X_sub, probs_sub, T_sub = X_expt[random_inds], probs[random_inds], T[random_inds]

        X_bin, y_bin = fitting.convertToBinaryClassifier(probs_sub, T_sub, X_sub)
        ybar = np.mean(y_bin)
        beta_null = np.log(ybar / (1 - ybar))
        null_weights = np.concatenate((np.array([beta_null]), np.zeros(X_sub.shape[-1])))
        nll_null = fitting.negLL_hotspot(null_weights, X_bin, y_bin, False, 'none', 0)

        subsample_weights.append(fitting.get_w(args.num_sites, X_bin, y_bin, nll_null)[0])

    subsample_weights = np.array(subsample_weights)
    np.save('subsampling-weights-{}-n{}-p{}-s{}.npy'.format(dataset, str(cell), str(p), str(args.subsample)), 
            subsample_weights)

elif args.resample is not None:
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
