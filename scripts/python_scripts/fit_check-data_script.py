import numpy as np
from scipy.io import loadmat, savemat
import os
import multielec_src.fitting as fitting
import multielec_src.multielec_utils as mutils
import statsmodels.api as sm
from copy import deepcopy

GSORT_BASE = '/Volumes/Scratch/Analysis'
ESTIM_ANALYSIS_BASE = '/Volumes/Lab/Users/praful/outputs/pp_out'
dataset = '2020-10-06-7'
estim = 'data003'
wnoise = 'kilosort_data000/data000'
p = 398
c = 296

data = loadmat(os.path.join(GSORT_BASE, dataset, estim, wnoise, f'fit_{dataset}_p{p}c{c}.mat'))

basename = f'/Volumes/Analysis/{dataset}/gsort'

# Load electrical data and g-sort data
outpath = os.path.join(basename, estim, wnoise)
parameters = loadmat(os.path.join(outpath, 'parameters.mat'))

cells = parameters['cells'].flatten()
patterns = parameters['patterns'].flatten()
num_cells = len(cells)
num_patterns = max(patterns)
num_movies = parameters['movies'].flatten()[0]

all_trials = np.array(np.memmap(os.path.join(outpath, 'trial.dat'),mode='r',shape=(num_patterns, num_movies), dtype='int16'), dtype=int)

amps_gsort = mutils.get_stim_amps_newlv(os.path.join(ESTIM_ANALYSIS_BASE, dataset, 'data005'), 
                                        len(all_trials))

params = data['params_true']
probs_fit = data['probs_fit'].flatten()
X = data['amps_fit']
subsample_fractions = data['subsample_fractions'].flatten()

# Find indices of rows in A corresponding to rows in B, preserving order
indices = [np.where(np.all(amps_gsort == row, axis=1))[0][0] for row in X]
remaining_inds = np.setdiff1d(np.arange(len(amps_gsort)), indices)
amps_remaining = deepcopy(amps_gsort[remaining_inds])

probs_flipped = np.zeros(len(amps_gsort))
probs_flipped[indices] = probs_fit

probs_remaining = fitting.sigmoidND_nonlinear(sm.add_constant(amps_remaining, has_constant='add'),
                                            params)
probs_flipped[remaining_inds] = np.where(probs_remaining > 0.5, 1, 0)
T = np.ones_like(probs_flipped) * 20

ms = [1, 2, 3, 4, 5]
zero_prob = 0.01
slope_bound = 100
R2_thresh = 0.025
reg_param = 0.5
method = 'L-BFGS-B'
reg_method = 'l2'

w_inits = []
for m in ms:
    w_init = np.array(np.random.normal(size=(m, amps_gsort.shape[1]+1)))
    z = 1 - (1 - zero_prob)**(1/len(w_init))
    w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
    w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
    w_inits.append(w_init)

opt, _ = fitting.fit_surface_earlystop(amps_gsort, probs_flipped, T, w_inits,
                            reg_method=reg_method, reg=[reg_param], slope_bound=slope_bound,
                            zero_prob=zero_prob, method=method,
                            R2_thresh=R2_thresh                           
)
params_fit, _, _ = opt
probs_pred = fitting.sigmoidND_nonlinear(sm.add_constant(amps_gsort, has_constant='add'), 
                                         params_fit)

MAE_full = np.mean(np.abs(probs_pred - probs_flipped))
RMSE_full = np.sqrt(np.mean((probs_pred - probs_flipped)**2))

print(f'MAE_full: {MAE_full}')
print(f'RMSE_full: {RMSE_full}')

NUM_RUNS = 100

RMSEs_all = []
MAEs_all = []

for run in range(NUM_RUNS):
    print(f'Run {run+1}/{NUM_RUNS}')
    w_inits = []
    for m in ms:
        w_init = np.array(np.random.normal(size=(m, amps_gsort.shape[1]+1)))
        z = 1 - (1 - zero_prob)**(1/len(w_init))
        w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
        w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
        w_inits.append(w_init)

    RMSEs_subsample = []
    MAEs_subsample = []
    for fraction in subsample_fractions:
        subsample_inds = np.random.choice(len(amps_gsort), int(fraction*len(amps_gsort)), replace=False)

        # Fit the model
        opt, _ = fitting.fit_surface_earlystop(amps_gsort[subsample_inds], probs_flipped[subsample_inds], T[subsample_inds], w_inits,
                                reg_method=reg_method, reg=[reg_param], slope_bound=slope_bound,
                                zero_prob=zero_prob, method=method,
                                R2_thresh=R2_thresh, verbose=False                         
        )
        params_subsample, _, R2_subsample = opt
        RMSE_subsample = np.sqrt(np.mean((probs_flipped - fitting.sigmoidND_nonlinear(sm.add_constant(amps_gsort, has_constant='add'), params_subsample))**2))
        RMSEs_subsample.append(RMSE_subsample)
        MAE_subsample = np.mean(np.abs(probs_flipped - fitting.sigmoidND_nonlinear(sm.add_constant(amps_gsort, has_constant='add'), params_subsample)))
        MAEs_subsample.append(MAE_subsample)
        print(f'Fraction: {fraction}, RMSE: {RMSE_subsample}, MAE: {MAE_subsample}')
    
    RMSEs_all.append(RMSEs_subsample)
    MAEs_all.append(MAEs_subsample)

RMSEs_all = np.array(RMSEs_all)
MAEs_all = np.array(MAEs_all)

savemat(os.path.join('/Volumes/Lab/Users/praful/multielectrode/figures/fig3', f'errors_{dataset}_p{p}c{c}_subsample-flipped.mat'), {'RMSEs_all': RMSEs_all,
    'MAEs_all': MAEs_all,
    'subsample_fractions': subsample_fractions,
    'RMSE_full': RMSE_full,
    'MAE_full': MAE_full,
    'probs_pred': probs_pred,
    'probs_flipped': probs_flipped})