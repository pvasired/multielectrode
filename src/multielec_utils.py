import numpy as np
from scipy.io import loadmat
import pickle
import os

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
                  p_thr=1/19, p_upper=1, downsample=False, downsample_trials=10, 
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
            
            p_thr = max(p_thr, 1/downsample_trials)
        
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
        
    good_inds = np.where((probs > p_thr) & (probs < p_upper) & (amplitudes[:, 0] < 1.5))[0]

    y = probs[good_inds]
    X = amplitudes[good_inds]
    T = trials[good_inds]

    return X, y, T