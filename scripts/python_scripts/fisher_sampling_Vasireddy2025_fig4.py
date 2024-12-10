import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
os.environ["CUDA_VISIBLE_DEVICES"]= '2'

import numpy as np
import multielec_src.fitting as fitting
import multielec_src.closed_loop as cl
from scipy.io import loadmat, savemat
import multiprocessing as mp
import statsmodels.api as sm
from copy import deepcopy, copy

def sample_spikes(p_true, t, error_rate_0=0, error_rate_1=0):
    """
    Helper function to sample spikes from a Bernoulli distribution.

    Parameters:
    p_true (np.ndarray AMPLITUDES X 1): True probabilities of spiking across amplitudes
                         for a given (cell, pattern)
    t (np.ndarray AMPLITUDES X 1): Number of trials across amplitudes for a given (cell, pattern)

    Returns:
    p_empirical_array (np.ndarray): Empirical probability of a spike across
                              amplitude for a given (cell, pattern)
    """
    p_true, t = np.array(p_true), np.array(t).astype(int)
    
    p_empirical = []
    for i in range(len(p_true)):
        # If there are no trials, set the empirical probability to 0.5
        if t[i] == 0:
            p_empirical += [0.5]
        
        # Else, sample from a Bernoulli distribution
        else:
            spikes = np.random.choice(np.array([0, 1]), 
                                                 p=np.array([(1-p_true[i])*(1-error_rate_0) + p_true[i]*error_rate_1, 
                                                             p_true[i]*(1-error_rate_1) + (1-p_true[i])*error_rate_0]), 
                                                 size=t[i])

            p_empirical += [np.mean(spikes)]
        
    p_empirical_array = np.array(p_empirical)

    return p_empirical_array

def sample_spikes_array(true_probs, trials, error_rate_0=0, error_rate_1=0,
                        NUM_THREADS=24):
    """
    Sample spikes across all cells and patterns using multiprocessing.

    Parameters:
    true_probs (np.ndarray CELLS X PATTERNS X AMPLITUDES): True probabilities of spikes
    trials (np.ndarray PATTERNS X AMPLITUDES): Number of trials
    NUM_THREADS (int): Number of threads to use for multiprocessing

    Returns:
    p_empirical_array (np.ndarray CELLS X PATTERNS X AMPLITUDES): Empirical probability of a spike across
                                    all cells and patterns
    """

    # Set up a list for multiprocessing
    input_list = []
    for i in range(len(true_probs)):
        for j in range(len(true_probs[i])):
            input_list += [(true_probs[i][j], trials[j], error_rate_0, error_rate_1)]
    
    # Run the multiprocessing
    pool = mp.Pool(processes=NUM_THREADS)
    results = pool.starmap_async(sample_spikes, input_list)
    mp_output = results.get()
    pool.close()

    return np.array(mp_output).reshape(true_probs.shape)

def get_performance_array(true_params, curr_probs, true_probs):
    
    error = 0
    cnt = 0
    for i in range(len(true_params)):
        for j in range(len(true_params[i])):
            if type(true_params[i][j]) != int:
                error += np.sqrt(np.sum((curr_probs[i][j] - true_probs[i][j])**2) / len(true_probs[i][j]))
                cnt += 1

    return error / cnt

GSORT_BASE = "/Volumes/Scratch/Analysis"
DATASET = "2020-10-18-0"
datarun = "data003"
wnoise = "kilosort_data000/data000"
cell = 85
pattern = 233

params = loadmat(os.path.join(GSORT_BASE, DATASET, datarun, wnoise, f'fit_{DATASET}_p{pattern}c{cell}.mat'))['params_true']
params_true = np.zeros((1, 1), dtype=object)
params_true[0][0] = params

ms = [1, 2, 3, 4, 5, 6]

amps_scan = np.array([np.array(np.meshgrid(np.linspace(-2, 2, 21), 
                                np.linspace(-2, 2, 21),
                                np.linspace(-2, 2, 21))).T.reshape(-1,3)])

probs_true_scan = np.zeros((1, 1, amps_scan.shape[1]))
probs_true_scan[0][0] = fitting.sigmoidND_nonlinear(
                        sm.add_constant(amps_scan[0], has_constant='add'), 
                        params_true[0][0])

def initialize_trials(T_prev, init_amps, init_trials):
    """Initialize trials with a given number of samples."""
    for i in range(len(T_prev)):
        init_inds = np.random.choice(np.arange(len(T_prev[i]), dtype=int), size=init_amps, replace=False)
        T_prev[i][init_inds] = init_trials
    return T_prev

# Initialization and trial parameters
init_trials = 2
init_amps = 100
trial_cap = 25

# Trial optimization parameters
budget = amps_scan.shape[0] * amps_scan.shape[1] * 0.05 
reg = None
T_step_size = 0.05
T_n_steps = 10000

# Explore/exploit parameters
num_iters = 10
rate = 0.75
exploit_factors = 1 - np.exp(-rate * np.arange(num_iters+1))

# Error rate parameters
error_rate_0 = 0
error_rate_1 = error_rate_0

# Fitting parameters
ms = [6]
ms_fit = [1, 2, 3, 4, 5, 6]
verbose = True
R2_cutoff = 0
prob_low = 0.2
bootstrapping = None
reg_method = 'l2'
method = 'L-BFGS-B'
regfit = 0
NUM_THREADS = 24
zero_prob = 0.01
slope_bound = 100

def fit_empirical_data(probs_empirical, amps_scan, T_prev, ms_fit, reg_method, reg_param, slope_bound, zero_prob, method, R2_thresh):
    probs_curr = np.zeros(probs_empirical.shape)
    for i in range(probs_empirical.shape[0]):
        for j in range(probs_empirical.shape[1]):
            w_inits = []

            for m in ms_fit:
                w_init = np.array(np.random.normal(size=(m, amps_scan[j].shape[1]+1)))
                z = 1 - (1 - zero_prob)**(1/len(w_init))
                w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
                w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
                w_inits.append(w_init)
                
            X = amps_scan[j]
            probs_fit = probs_empirical[i][j]
            T = T_prev[j]
            opt, _ = fitting.fit_surface_earlystop(X, probs_fit, T, w_inits,
                                    reg_method=reg_method, reg=[reg_param], slope_bound=slope_bound,
                                    zero_prob=zero_prob, method=method,
                                    R2_thresh=R2_thresh)
            params_fit, _, _ = opt
            probs_curr[i][j] = fitting.sigmoidND_nonlinear(
                                    sm.add_constant(amps_scan[j], has_constant='add'),
                                    params_fit)
    
    return probs_curr

NUM_RUNS = 100

all_performances = []
all_performances_uniform = []
all_num_samples = []
all_num_samples_uniform = []
all_trials = []

for iteration in range(NUM_RUNS):
    print(f"Run {iteration+1}")
    T_prev = np.zeros((amps_scan.shape[0], amps_scan.shape[1]), dtype=float)
    T_prev = initialize_trials(T_prev, init_amps, init_trials)
    T_prev_uniform = deepcopy(T_prev)

    probs_empirical = sample_spikes_array(probs_true_scan, T_prev, error_rate_0=error_rate_0, error_rate_1=error_rate_1, NUM_THREADS=24)
    probs_empirical_uniform = deepcopy(probs_empirical)

    performances = []
    performances_uniform = []
    num_samples = []
    num_samples_uniform = []

    iter_cnt = 0
    while True:
        exploit_factor = exploit_factors[iter_cnt]
        T_new, _, t_final, _, _ = cl.fisher_sampling_1elec(
                                        probs_empirical, 
                                        T_prev, amps_scan,
                                        T_step_size=T_step_size,
                                        T_n_steps=T_n_steps,
                                        verbose=verbose, budget=budget, ms=ms, reg=reg,
                                        return_probs=True,
                                        R2_cutoff=R2_cutoff,
                                        min_prob=prob_low,
                                        exploit_factor=exploit_factor, 
                                        trial_cap=trial_cap,
                                        bootstrapping=bootstrapping,
                                        X_all=amps_scan[0],
                                        reg_method=reg_method,
                                        regfit=[regfit],
                                        NUM_THREADS=NUM_THREADS,
                                        slope_bound=slope_bound)
        
        probs_curr = fit_empirical_data(probs_empirical, amps_scan, T_prev, ms_fit, reg_method, regfit, slope_bound, zero_prob, method, R2_cutoff)
        performance = get_performance_array(params_true, probs_curr, probs_true_scan)
        if iter_cnt == 0:
            performance_uniform = performance
        else:
            probs_curr_uniform = fit_empirical_data(probs_empirical_uniform, amps_scan, T_prev_uniform, ms_fit, reg_method, regfit, slope_bound, zero_prob, method, R2_cutoff)
            performance_uniform = get_performance_array(params_true, probs_curr_uniform, probs_true_scan)
        
        if verbose:
            print(performance, performance_uniform)
        
        performances.append(performance)
        performances_uniform.append(performance_uniform)
        
        num_samples.append(np.sum(T_prev))
        num_samples_uniform.append(np.sum(T_prev_uniform))

        iter_cnt += 1

        if iter_cnt > num_iters:
            break

        p_new = sample_spikes_array(probs_true_scan, T_new, error_rate_0=error_rate_0, error_rate_1=error_rate_1, NUM_THREADS=24)
        p_tmp = (p_new * T_new[np.newaxis, :, :] + probs_empirical * T_prev[np.newaxis, :, :]) / ((T_new + T_prev)[np.newaxis, :, :])
        T_tmp = T_new + T_prev

        p_tmp = np.nan_to_num(p_tmp, nan=0.5)

        probs_empirical = p_tmp
        T_prev = T_tmp
        
        random_extra = np.random.choice(len(T_new.flatten()), size=int(np.sum(T_new)), replace=True)
        T_new_uniform = np.array(np.bincount(random_extra, minlength=len(T_new.flatten())).astype(int).reshape(T_new.shape), dtype=float)
        p_new_uniform = sample_spikes_array(probs_true_scan, T_new_uniform, error_rate_0=error_rate_0, error_rate_1=error_rate_1, NUM_THREADS=24)

        p_tmp_uniform = (p_new_uniform * T_new_uniform[np.newaxis, :, :] + probs_empirical_uniform * T_prev_uniform[np.newaxis, :, :]) / ((T_prev_uniform + T_new_uniform)[np.newaxis, :, :])
        T_tmp_uniform = T_prev_uniform + T_new_uniform

        p_tmp_uniform = np.nan_to_num(p_tmp_uniform, nan=0.5)

        probs_empirical_uniform = p_tmp_uniform
        T_prev_uniform = T_tmp_uniform
    
    all_performances.append(performances)
    all_performances_uniform.append(performances_uniform)
    all_num_samples.append(num_samples)
    all_num_samples_uniform.append(num_samples_uniform)
    all_trials.append(T_prev)

all_performances = np.array(all_performances)
all_performances_uniform = np.array(all_performances_uniform)
all_num_samples = np.array(all_num_samples)
all_num_samples_uniform = np.array(all_num_samples_uniform)
all_trials = np.array(all_trials)

# Save results to .mat file
savemat(f"fisher_sampling_Vasireddy2025_fig4_{DATASET}_p{pattern}c{cell}.mat", 
        {'all_performances': all_performances,
         'all_performances_uniform': all_performances_uniform,
         'all_num_samples': all_num_samples,
         'all_num_samples_uniform': all_num_samples_uniform,
         'all_trials': all_trials})