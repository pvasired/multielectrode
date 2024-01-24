import numpy as np
from scipy.io import loadmat
import pickle
import os
import multielec_src.fitting as fitting
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
    """
    Get the time-collapsed EI of a cell
    
    Parameters:
    vcd (object): visionloader object for the dataset
    cell_no (int): Cell number for the target cell EI
    thr_factor (float): value for which to threshold EI for the cell
    
    Returns:
    good_inds (np.ndarray): Indices of electrodes where EI meets threshold
    collapsed_EI: Time-collapsed EI according to minimum value across time,
                  absolute valued
    
    """
    # Read the EI for a given cell
    cell_ei = vcd.get_ei_for_cell(cell_no).ei
    
    # Collapse into maximum value
    collapsed_ei = np.amax(np.absolute(cell_ei), axis=1)
    
    channel_noise = vcd.channel_noise
    
    # Threshold the EI to pick out only electrodes with large enough values
    good_inds = np.argwhere(np.abs(collapsed_ei) > thr_factor * channel_noise).flatten()
    
    return good_inds, np.abs(collapsed_ei)

def get_stim_elecs_newlv(analysis_path, pattern):
    """
    Read a newlv pattern files directory and get the stimulation 
    electrodes.
    
    Parameters:
    analysis_path: Path to preprocessed data
    pattern: Pattern for which to get stimulation electrodes

    Returns:
    stimElecs (np.ndarray): Stimulation electrodes for the pattern
    """
    patternStruct = loadmat(os.path.join(analysis_path, "pattern_files/p" + str(pattern) + ".mat"), struct_as_record=False, squeeze_me=True)['patternStruct']
    return patternStruct.stimElecs

def get_stim_amps_newlv(analysis_path, pattern):
    """
    Read a newlv pattern files directory and get the stimulation 
    amplitudes.
    
    Parameters:
    analysis_path: Path to preprocessed data
    pattern: Pattern for which to get stimulation smplitudes

    Returns:
    amplitudes (np.ndarray): Stimulation amplitudes for the pattern
    """
    patternStruct = loadmat(os.path.join(analysis_path, "pattern_files/p" + str(pattern) + ".mat"), struct_as_record=False, squeeze_me=True)['patternStruct']
    return patternStruct.amplitudes
    
def triplet_cleaning(X_expt_orig, probs_orig, T_orig, n_neighbors=6, n=10, k=2, percentile=1.0,
                     num_points=20, return_inds=False, NUM_THREADS=24, dist_thr=0.3):
    
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
    pool = mp.Pool(processes=NUM_THREADS)
    results = pool.starmap_async(fitting.enforce_3D_monotonicity, product(np.arange(len(X_expt_orig), dtype=int).tolist(), 
                                                                  [X_expt_orig], [probs_orig], [k], [percentile],
                                                                  [num_points], [dist_thr]))
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

    removed_inds = []
    for i in range(len(X_expt_dirty)):
        neighbors = np.setdiff1d(np.where(dists[i] <= dist_thr)[0], i)
        # neighbors = np.argsort(dists[i])[1:n_neighbors+1]

        if len(neighbors) == 0:
            X_clean.append(X_expt_dirty[i])
            p_clean.append(probs_dirty[i])
            T_clean.append(T_dirty[i])

            continue

        mean = np.mean(probs_dirty[neighbors])
        stdev = np.std(probs_dirty[neighbors])

        if probs_dirty[i] > mean + n * stdev or probs_dirty[i] < mean -  n * stdev:
            removed_inds.append(good_inds[i])
            continue
        else:
            X_clean.append(X_expt_dirty[i])
            p_clean.append(probs_dirty[i])
            T_clean.append(T_dirty[i])

    X_clean = np.array(X_clean)
    p_clean = np.array(p_clean)
    T_clean = np.array(T_clean)

    good_inds_tot = np.setdiff1d(good_inds, np.array(removed_inds))

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

    if return_inds:
        return good_inds_tot
    
    else:
        return X_clean, p_clean, T_clean