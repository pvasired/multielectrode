import numpy as np
from scipy.io import loadmat
import os

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