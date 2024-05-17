import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
import matplotlib.pyplot as plt
import multielec_src.multielec_utils as mutils
import multielec_src.fitting as fitting
from scipy.io import loadmat
from copy import deepcopy
import statsmodels.api as sm
import visionloader as vl

# Read APL sorter data
data = np.genfromtxt('/Volumes/Scratch/Users/praful/2020_macaque_trips_data005.csv', delimiter=',',
                     skip_header=1)

def combine_gsort_oases(amps_gsort, probs_gsort, amps_oases, probs_oases, spont_limit=0.15):
    probs_combined = []
    amps_combined = []
    inds_combined = []
    for i in range(len(amps_gsort)):
        if dataset == '2020-10-18-5':
            if amps_gsort[i][0] > 1.5 or amps_gsort[i][2] > 1.5:
                continue
                
        if probs_gsort[i] <= spont_limit:
            oases_ind = np.where(np.all(amps_oases == amps_gsort[i], axis=1))[0]
            if len(oases_ind) == 0 or len(oases_ind) > 1:
                raise ValueError(f'Could not find oases amp for gsort amp {amps_gsort[i]}')
            
            if probs_oases[oases_ind[0]] == 0:#or probs_oases[oases_ind[0]] == 1:
                probs_combined.append(probs_oases[oases_ind[0]])
                amps_combined.append(amps_gsort[i])
                inds_combined.append(i)
        
        else:
            probs_combined.append(probs_gsort[i])
            amps_combined.append(amps_gsort[i])
            inds_combined.append(i)

    return np.array(probs_combined), np.array(amps_combined), np.array(inds_combined)

# Read g-sort data
GSORT_BASE = "/Volumes/Scratch/Analysis"
WNOISE_ANALYSIS_BASE = "/Volumes/Analysis"
ESTIM_ANALYSIS_BASE = "/Volumes/Analysis"
dataset = "2020-10-06-7"
wnoise = "kilosort_data000/data000"
estim_neg = "data005"
PSEUDO_ESF_PATH = f'/Volumes/Scratch/Users/praful-tmp/oases/{dataset}/{estim_neg}.pseudo_esf'
pseudo_esf = np.fromfile(PSEUDO_ESF_PATH, dtype=np.int32).reshape(-1, 3, order='F')

vstim_datapath = os.path.join(WNOISE_ANALYSIS_BASE, dataset, wnoise)
vstim_datarun = os.path.basename(os.path.normpath(vstim_datapath))
vcd = vl.load_vision_data(vstim_datapath, vstim_datarun,
                          include_neurons=True,
                          include_ei=True,
                          include_params=True,
                          include_noise=True)
vcd.update_cell_type_classifications_from_text_file(os.path.join(vstim_datapath, 'data000.classification_agogliet.txt'))
coords = vcd.electrode_map

# Load electrical data and g-sort data
outpath = os.path.join(GSORT_BASE, dataset, estim_neg, wnoise)
electrical_path = os.path.join(ESTIM_ANALYSIS_BASE, dataset, estim_neg)
parameters = loadmat(os.path.join(outpath, 'parameters.mat'))

cells = parameters['cells'].flatten()
patterns = parameters['patterns'].flatten()
num_cells = len(cells)
num_patterns = max(patterns)
num_movies = parameters['movies'].flatten()[0]
gsorted_cells = parameters['gsorted_cells'].flatten()

all_probs = np.nan_to_num(np.array(np.memmap(os.path.join(outpath, 'init_probs.dat'),
                                            mode='r',shape=(num_cells, num_patterns, num_movies), dtype='float32')))
all_trials = np.array(np.memmap(os.path.join(outpath, 'trial.dat'),mode='r',shape=(num_patterns, num_movies), dtype='int16'), dtype=int)

# TODO: change this to parasol and midget if monkey

gsorted_cells_new = []
for i in range(len(gsorted_cells)):
    if 'parasol' in vcd.get_cell_type_for_cell(cells[gsorted_cells[i]]).lower():
        gsorted_cells_new.append(gsorted_cells[i])
    # if 'midget' in vcd.get_cell_type_for_cell(cells[gsorted_cells[i]]).lower():
    #     gsorted_cells_new.append(gsorted_cells[i])

gsorted_cells = np.array(gsorted_cells_new)

amps_plot = np.array(np.meshgrid(np.linspace(-2, 2, 21), 
                                np.linspace(-2, 2, 21),
                                np.linspace(-2, 2, 21))).T.reshape(-1,3)

min_inds_oases = 50
min_inds_gsort = 50
spont_limit = 0.15
bootstrapping = None
ms = [1, 2, 3, 4, 5, 6]
zero_prob = 0.01
slope_bound = 100
R2_thresh = 0.01
l2reg = 0.5
method = 'L-BFGS-B'
reg_method = 'l2'

amp_vals = np.concatenate((np.arange(0, 43), -np.arange(1, 43)))*3/128*4
amps = np.zeros((len(amp_vals)**3, 3))
ii = 0 
for a1 in amp_vals:
    for a2 in amp_vals:
        for a3 in amp_vals:
            amps[ii] = [a1, a2, a3]
            ii += 1

subsample_fractions = [1e-3, 3e-3, 5e-3, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
N = 1000

def find_indices(arr1, arr2):
    indices = []
    for row in arr1:
        idx = np.where((arr2 == row).all(axis=1))[0]
        if idx.size > 0:
            indices.append(idx[0])
        else:
            raise ValueError(f'Could not find index for {row}')
    return np.array(indices)

for k in range(len(gsorted_cells)):
    i = gsorted_cells[k]
    cell = cells[i]
    for j in range(len(patterns)):
        print(f'Cell {k+1}/{len(gsorted_cells)}, Pattern {j+1}/{len(patterns)}')
        p = patterns[j]
        stim_elecs = mutils.get_stim_elecs_newlv(electrical_path, p)

        # Index APL sorter data
        cell_data = data[np.where((data[:, 0].astype(int) == cell) & (data[:, 2].astype(int) == p))[0]]

        amps_gsort = mutils.get_stim_amps_newlv(electrical_path, p)
        probs_gsort = all_probs[np.where(cells == cell)[0][0], p-1, :]

        amp_inds = np.unique(pseudo_esf[np.where(pseudo_esf[:, 1] == p)[0], 2])
        probs = np.zeros(len(amp_vals)**3)

        current_levels, counts = np.unique(cell_data[:, 3], return_counts=True)
        current_levels = current_levels.astype(int)

        gsort_inds = find_indices(amps[current_levels], amps_gsort)
        probs[current_levels] = counts / all_trials[p-1][gsort_inds]

        amps_oases = amps[amp_inds]
        probs_oases = probs[amp_inds]

        if len(probs_oases[probs_oases >= spont_limit]) > min_inds_oases and len(probs_gsort[probs_gsort >= spont_limit]) > min_inds_gsort:
            print("Cell: {}, Pattern: {}".format(cell, p))

            combined_probs, combined_amps, combined_inds = combine_gsort_oases(amps_gsort, probs_gsort, amps_oases, probs_oases)

            w_inits = []
            for m in ms:
                w_init = np.array(np.random.normal(size=(m, combined_amps.shape[1]+1)))
                z = 1 - (1 - zero_prob)**(1/len(w_init))
                w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
                w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
                w_inits.append(w_init)

            X, probs_fit, trials = deepcopy(combined_amps), deepcopy(combined_probs), deepcopy(all_trials[p-1][combined_inds])
            opt, _ = fitting.fit_surface(X, probs_fit, trials, w_inits, bootstrapping=bootstrapping, X_all=amps_plot,
                                        reg_method=reg_method, reg=[l2reg], slope_bound=slope_bound,
                                        zero_prob=zero_prob, R2_thresh=R2_thresh, method=method                             
            )
            params_true, _, R2_full = opt
            print(params_true, R2_full)
            probs_pred_expt = fitting.sigmoidND_nonlinear(sm.add_constant(X, has_constant='add'),
                                                                params_true)
            
            RMSE_all = np.sqrt(np.mean((probs_fit - probs_pred_expt)**2))
            print(f'RMSE: {RMSE_all}')

            skip = input('Skip? (y/n): ')
            if skip == 'y':
                continue
                        
            R2s_all = np.zeros((len(subsample_fractions), N))
            RMSEs_all = np.zeros((len(subsample_fractions), N))
            for fraction in subsample_fractions:
                print(f'Training Size: {fraction}')
                R2s = np.zeros(N)
                RMSEs = np.zeros(N)
                for iteration in range(N):
                    print(f'Bootstrap Iteration: {iteration+1}/{N}')
                    X_sub = deepcopy(amps_gsort)
                    probs_sub = np.zeros(len(amps_gsort))
                    T_sub = np.zeros(len(amps_gsort))

                    random_inds = np.random.choice(len(amps_gsort), int(fraction*np.sum(all_trials[p-1])), replace=True)

                    for ind in random_inds:
                        if ind in combined_inds:
                            probs_sub[ind] += np.random.choice([0, 1], p=[1 - combined_probs[np.where(combined_inds == ind)[0][0]], 
                                                                          combined_probs[np.where(combined_inds == ind)[0][0]]])
                            T_sub[ind] += 1
                    
                    X_sub = X_sub[np.where(T_sub > 0)[0]]
                    probs_sub = probs_sub[np.where(T_sub > 0)[0]]
                    T_sub = T_sub[np.where(T_sub > 0)[0]]

                    probs_sub = probs_sub / T_sub
                    
                    w_inits = []
                    for m in ms:
                        w_init = np.array(np.random.normal(size=(m, X_sub.shape[1]+1)))
                        z = 1 - (1 - zero_prob)**(1/len(w_init))
                        w_init[:, 0] = np.clip(w_init[:, 0], None, np.log(z/(1-z)))
                        w_init[:, 1:] = np.clip(w_init[:, 1:], -slope_bound, slope_bound)
                        w_inits.append(w_init)
                        
                    opt, _ = fitting.fit_surface(X_sub, probs_sub, T_sub, w_inits, bootstrapping=bootstrapping, X_all=amps_plot,
                                                reg_method=reg_method, reg=[l2reg], slope_bound=slope_bound,
                                                zero_prob=zero_prob, R2_thresh=R2_thresh, method=method                          
                    )
                    params_sub, _, R2 = opt
                    probs_pred_sub = fitting.sigmoidND_nonlinear(sm.add_constant(X, has_constant='add'),
                                                                params_sub)
                    RMSE = np.sqrt(np.mean((probs_fit - probs_pred_sub)**2))

                    R2s[iteration] = R2
                    RMSEs[iteration] = RMSE
                    print(f'R2: {R2}, RMSE: {RMSE}, Training Size: {fraction}')
                
                R2s_all[subsample_fractions.index(fraction)] = R2s
                RMSEs_all[subsample_fractions.index(fraction)] = RMSEs
            
            np.save(f'bootstrap_results_{dataset}_{cell}_{p}.npy', RMSEs_all)
            # Calculate the mean and standard error along the columns
            means = np.mean(RMSEs_all, axis=1)
            std_error = np.std(RMSEs_all, axis=1) / np.sqrt(RMSEs_all.shape[1])

            # Confidence interval (95%)
            confidence_interval = 1.96 * std_error

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(subsample_fractions, means, label='Mean')
            plt.fill_between(subsample_fractions, (means - confidence_interval), (means + confidence_interval), color='b', alpha=0.2, label='95% Confidence Interval')
            plt.xscale('log')
            plt.title('Mean and 95% Confidence Interval of Measurements')
            plt.xlabel('Subsample Fraction')
            plt.ylabel('RMSE')
            plt.legend()
            plt.savefig(f'bootstrap_results_{dataset}_{cell}_{p}.png')
            plt.show()