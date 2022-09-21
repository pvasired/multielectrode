import numpy as np
from scipy.io import loadmat
import pickle
import os
import src.fitting as fitting
from scipy.optimize import minimize
import statsmodels.api as sm
from itertools import product
import multiprocessing as mp
from scipy.spatial.distance import cdist

# Current values in uA

Ivals = np.array([0.10053543, 0.11310236, 0.11938583, 0.13195276, 0.14451969,                        
                       0.16337008, 0.17593701, 0.1947874 , 0.2136378 , 0.23877165,
                       0.25762205, 0.2780315 , 0.30330709, 0.35385827, 0.37913386,
                       0.42968504, 0.45496063, 0.50551181, 0.55606299, 0.60661417,
                       0.68244094, 0.73299213, 0.8088189 , 0.88464567, 0.98574803,
                       1.10433071, 1.20472441, 1.30511811, 1.40551181, 1.60629921,
                       1.70669291, 1.90748031, 2.10826772, 2.30905512, 2.50984252,
                       2.81102362, 3.11220472, 3.41338583, 3.71456693, 4.1161])

def get_collapsed_ei_thr(vcd, cell_no, thr_factor):
    # Read the EI for a given cell
    cell_ei = vcd.get_ei_for_cell(cell_no).ei
    
    # Collapse into maximum value
    collapsed_ei = np.amin(cell_ei, axis=1)
    
    channel_noise = vcd.channel_noise
    
    # Threshold the EI to pick out only electrodes with large enough values
    good_inds = np.argwhere(np.abs(collapsed_ei) > thr_factor * channel_noise).flatten()
    
    return good_inds, np.abs(collapsed_ei)

def get_stim_elecs_newlv(analysis_path, pattern):
    patternStruct = loadmat(os.path.join(analysis_path, "pattern_files/p" + str(pattern) + ".mat"), struct_as_record=False, squeeze_me=True)['patternStruct']
    return patternStruct.stimElecs

def get_stim_amps_newlv(analysis_path, pattern):
    patternStruct = loadmat(os.path.join(analysis_path, "pattern_files/p" + str(pattern) + ".mat"), struct_as_record=False, squeeze_me=True)['patternStruct']
    return patternStruct.amplitudes

def loadNewLVData(electrical_path, gsort_path, dataset, estim, wnoise, p, n,
                  p_thr=-1, p_upper=1, downsample=False, downsample_trials=10, 
                  downsample_factor=2, load_from_mat=False, MATFILE_BASE=''):
    
    if load_from_mat:
        cell_matpath = os.path.join(MATFILE_BASE, dataset, estim, wnoise, "p" + str(p), "n" + str(n) + ".mat")
        data_dict = loadmat(cell_matpath)
        amplitudes = data_dict['amplitudes']
        probs = data_dict['probabilities'].flatten()
        trials = data_dict['trials'].flatten()
        
        if downsample:
            allowed_amps = np.unique(amplitudes[:, 0])[::downsample_factor]
            allowed_inds = np.where(np.all(np.isin(amplitudes, allowed_amps), axis=1))[0]

            amplitudes = amplitudes[allowed_inds]
            dprobs = np.zeros(len(allowed_inds))
            dtrials = np.zeros(len(allowed_inds), dtype=int)
            for i in range(len(allowed_inds)):
                k = allowed_inds[i]
                num1s = int(probs[k] * trials[k])
                num0s = trials[k] - num1s
                spikes = np.concatenate((np.ones(num1s), np.zeros(num0s)))
                num_samples = min(downsample_trials, trials[k])
                
                sampled_spikes = np.random.choice(spikes, size=num_samples, replace=False)
                dprobs[i] = np.sum(sampled_spikes) / num_samples
                dtrials[i] = num_samples
                
            probs = dprobs
            trials = dtrials
                    
    else:
        filepath = os.path.join(gsort_path, 
                                dataset, estim, wnoise, "p" + str(p))

        amplitudes = get_stim_amps_newlv(electrical_path, p)
        if downsample:
            allowed_amps = np.unique(amplitudes[:, 0])[::downsample_factor]
            allowed_inds = np.where(np.all(np.isin(amplitudes, allowed_amps), axis=1))[0]

        else:
            allowed_inds = np.arange(len(amplitudes), dtype=int)

        amplitudes = amplitudes[allowed_inds]
        num_pts = len(allowed_inds)

        probs = np.zeros(num_pts)
        trials = np.zeros(num_pts, dtype=int)

        for i in range(len(allowed_inds)):
            k = allowed_inds[i]
            with open(os.path.join(filepath, "gsort_newlv_v2_n" + str(n) + "_p" + str(p) + "_k" + str(k) + ".pkl"), "rb") as f:
                prob_dict = pickle.load(f)
                prob = prob_dict["cosine_prob"][0]
                num_trials = prob_dict["num_trials"]

                if downsample:
                    num1s = int(prob * num_trials)
                    num0s = num_trials - num1s

                    spikes = np.concatenate((np.ones(num1s), np.zeros(num0s)))
                    num_samples = min(downsample_trials, num_trials)
                    sampled_spikes = np.random.choice(spikes, size=num_samples, replace=False)

                    probs[i] = np.sum(sampled_spikes) / num_samples
                    trials[i] = num_samples

                else:
                    probs[i] = prob
                    trials[i] = num_trials

        if downsample:
            p_thr = max(p_thr, 1/downsample_trials)
        
    if dataset == '2020-10-18-5':
        good_inds = np.where((probs >= p_thr) & (probs < p_upper) & (amplitudes[:, 0] < 1.5) & (np.any(np.absolute(amplitudes) > 0, axis=1)))[0]
       
    else:
        good_inds = np.where((probs >= p_thr) & (probs < p_upper) & (np.linalg.norm(amplitudes, axis=1) > 0))[0]

    y = probs[good_inds]
    y[y == 0] = 1e-5
    X = amplitudes[good_inds]
    T = trials[good_inds]

    return X, y, T

def get1elecCurve(dataset, gsort_path_1elec, estim_1elec, wnoise, p, n, spont_limit=0.2, noise_limit=0.1,
                  curr_min=0.1, curr_max=4, curr_points=1000, return_params=False, zero_prob=0.01):

    currs = np.linspace(curr_min, curr_max, curr_points)
    elec = p
    filepath_1elec = os.path.join(gsort_path_1elec, dataset, estim_1elec, wnoise, "p" + str(elec))

    k = 0
    probs = []
    trials = []
    while True:
        try:
            with open(os.path.join(filepath_1elec, "gsort_single_v2_n" + str(n) + "_p" + str(elec) + "_k" + str(k) + ".pkl"), "rb") as f:
                prob_dict = pickle.load(f)
                probs.append(prob_dict["cosine_prob"][0])
                trials.append(prob_dict["num_trials"])

        except:
            break

        k += 1

    trials = np.array(trials, dtype=int)
    probs = np.array(probs)
    probs[:12] = 0
    probs[probs < noise_limit] = 0
    probs = fitting.disambiguate_sigmoid(probs, spont_limit=spont_limit, noise_limit=noise_limit)

    bounds = [(None, np.log(zero_prob / (1 - zero_prob))),
              (0, 20)]
    X_bin, y_bin = fitting.convertToBinaryClassifier(probs, trials, Ivals[:k].reshape(-1, 1))
    results = minimize(fitting.negLL, x0=np.array([-1, 1]), args=(X_bin, y_bin, False, 'none'),
                       bounds=bounds)

    sigmoid = fitting.fsigmoid(sm.add_constant(currs.reshape(-1, 1)), results.x)

    if not return_params:
        return currs, Ivals, sigmoid, probs
    
    else:
        return currs, Ivals, sigmoid, probs, results.x
    
def triplet_cleaning(X_expt_orig, probs_orig, T_orig, electrical_path, p, dir_thr=0.1,
                     n_neighbors=6, n=2, radius=6, high_thr=0.9, low_thr=0.1, prob_buffer=1e-5, num_trials=20):
    
    # X_scan = get_stim_amps_newlv(electrical_path, p)

    # pool = mp.Pool(processes=48)
    # results = pool.starmap_async(fitting.enforce_3D_monotonicity, product(np.arange(len(X_expt_orig), dtype=int).tolist(), 
    #                                                             [X_expt_orig], [probs_orig], [T_orig]))
    # output = results.get()
    # pool.close()

    # mono_data = np.array(output, dtype=object)[np.where(np.equal(np.array(output, dtype=object), None) == False)[0]]

    # Xmono = np.zeros((len(mono_data), 3))
    # ymono = np.zeros(len(mono_data))
    # Tmono = np.zeros(len(mono_data), dtype=int)

    # for i in range(len(mono_data)):
    #     Xmono[i] = mono_data[i][0]
    #     ymono[i] = mono_data[i][1]
    #     Tmono[i] = mono_data[i][2]

    # Line enforcement of monotonicity
    pool = mp.Pool(processes=48)
    results = pool.starmap_async(fitting.enforce_3D_monotonicity, product(np.arange(len(X_expt_orig), dtype=int).tolist(), 
                                                                  [X_expt_orig], [probs_orig]))
    output = results.get()
    pool.close()

    good_inds = np.where(np.array(output))[0]
    X_expt_dirty = X_expt_orig[good_inds]
    probs_dirty = probs_orig[good_inds]
    T_dirty = T_orig[good_inds]

    # Local outlier cleaning

    dists = cdist(X_expt_dirty, X_expt_dirty)
    X_clean = []
    p_clean = []
    T_clean = []
    for i in range(len(X_expt_dirty)):
        neighbors = np.argsort(dists[i])[1:n_neighbors+1]
        mean = np.mean(probs_dirty[neighbors])
        stdev = np.std(probs_dirty[neighbors])

        if probs_dirty[i] > mean + n * stdev or probs_dirty[i] < mean -  n * stdev:
            continue
        else:
            X_clean.append(X_expt_dirty[i])
            p_clean.append(probs_dirty[i])
            T_clean.append(T_dirty[i])

    X_clean = np.array(X_clean)
    p_clean = np.array(p_clean)
    T_clean = np.array(T_clean)

    # X_expt_dirty, probs_dirty, T_dirty = fitting.enforce_3D_monotonicity(X_expt_orig, probs_orig, T_orig)
    

    # # 0/1 pinning based on local mean probability

    # dists = cdist(X_clean, X_scan)
    # dists_clean = cdist(X_clean, X_clean)
    # matching_inds = np.array(np.all((X_scan[:,None,:]==X_clean[None,:,:]),axis=-1).nonzero()).T

    # X_new = []
    # probs_new = []
    # for i in range(len(dists)):
    #     mean_prob = np.mean(p_clean[np.argsort(dists_clean[i])[:n_neighbors]])
    #     if mean_prob >= high_thr:
    #         neighbors = np.argsort(dists[i])[1:radius+1]
    #         new_points = np.setdiff1d(neighbors, matching_inds[:, 0])
    #         X_new.append(X_scan[new_points])
    #         probs_new.append(np.ones(len(new_points)) - prob_buffer)

    #     elif mean_prob <= low_thr:
    #         neighbors = np.argsort(dists[i])[1:radius+1]
    #         new_points = np.setdiff1d(neighbors, matching_inds[:, 0])
    #         X_new.append(X_scan[new_points])
    #         probs_new.append(np.zeros(len(new_points)) + prob_buffer)

    # X_new, idx = np.unique(np.vstack(X_new), axis=0, return_index=True)
    # probs_new = np.hstack(probs_new)[idx]
    # T_new = np.ones(len(probs_new), dtype=int) * num_trials

    # X_expt = np.vstack((X_clean, X_new))
    # probs = np.hstack((p_clean, probs_new))
    # T = np.hstack((T_clean, T_new))
    
    # good_inds = np.where(probs_orig >= 0.2)[0]

    return X_clean, p_clean, T_clean