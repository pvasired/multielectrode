import numpy as np
from tqdm import tqdm
import scipy.signal
from scipy.io import loadmat, savemat
import visionloader as vl
from joblib import Memory
import os
import shutil
memory = Memory(os.getcwd())

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Dataset in format YYYY-MM-DD-P.')
parser.add_argument('-w', '--wnoise', type=str, help='White noise run in format dataXXX (or streamed/dataXXX or kilosort_dataXXX/dataXXX, etc.).')
parser.add_argument('-e', '--estim', type=str, help='Estim run in format dataXXX.')
parser.add_argument('-o', '--output', type=str, help='/path/to/output/directory.')

args = parser.parse_args()

VISUAL_ANALYSIS_BASE = '/Volumes/Analysis'
ESTIM_ANALYSIS_BASE = '/Volumes/Analysis'
CORR_THR = 0.95

DATASET = args.dataset
ESTIM_DATARUN = args.estim
VSTIM_DATARUN_2 = args.wnoise
write_path = args.output

def with_ttl(raw_data:np.ndarray):
    """
    Asserts that exactly one dimension of the provided raw data has a size of 512 or 519.
    It then returns a tensor with that dimension expanded by +1, and the 0th index filled in with all 0s
    """
    is_512 = 512 in raw_data.shape
    if is_512:
        assert 519 not in raw_data.shape, f'raw_data must have a dimension of size 512 or 519, but has shape {raw_data.shape}'
        critical_axis = np.where(np.array(raw_data.shape) == 512)[0][0]
    else:
        assert 519 in raw_data.shape, f'raw_data must have a dimension of size 512 or 519, but has shape {raw_data.shape}'
        critical_axis = np.where(np.array(raw_data.shape) == 519)[0][0]
    append_data_shape = list(raw_data.shape)
    append_data_shape[critical_axis] = 1
    append_data = np.zeros(append_data_shape)
    return np.concatenate((append_data, raw_data), axis=critical_axis)

def correlate_ei_tensors(t1:np.ndarray, t2:np.ndarray):
    assert t1.ndim == 3, "T1 must be 3D"
    assert t2.ndim == 3, "T2 must be 3D"
    assert t1.shape[2] == t2.shape[2], "T1 and T2 must have the same number of channels"
    num_t1 = t1.shape[0]
    num_t2 = t2.shape[0]
    num_t2_time_samples = t2.shape[1]
    num_channels = t1.shape[2]
    t1_norms = np.linalg.norm(t1, axis=(1, 2))
    t2_norms = np.linalg.norm(t2, axis=(1, 2))
    assert t1_norms.size == num_t1, "T1 norms must have the same number of elements as t1"
    correlation_matrix = np.zeros((num_t1, num_t2))
    for t1_index in range(num_t1):
        t1_sub_matrix = t1[t1_index, :, :]
        zero_pad = np.zeros((num_t2_time_samples - 1, num_channels))
        t1_sub_matrix = np.concatenate((zero_pad, t1_sub_matrix, zero_pad), axis=0)
        assert t1_sub_matrix.shape[1] == num_channels, "T1 sub matrix must have the same number of channels as t1"
        for t2_index in tqdm(range(num_t2), desc=f'Correlating t1 index {t1_index} out of {num_t1} with all t2 indices', total=num_t2):
            correlation_matrix[t1_index, t2_index] = scipy.signal.correlate2d(t1_sub_matrix, t2[t2_index, :, :], mode='valid').max()/t1_norms[t1_index]/t2_norms[t2_index]
    return correlation_matrix

@memory.cache(ignore=['analysis_base'])
def match_eis_across_wn_519(piece:str, wn_one:str, wn_two:str, cell_ids_one, cell_ids_two, analysis_base='/pool0/lotlikar/Analysis'):
    assert len(cell_ids_one) <= len(cell_ids_two), "Cell ids one must be less than or equal to cell ids two"
    vstim_data_one = vl.load_vision_data(f"{analysis_base}/{piece}/{wn_one}/", wn_one.split("/")[-1], include_ei=True)
    vstim_data_two = vl.load_vision_data(f"{analysis_base}/{piece}/{wn_two}/", wn_two.split("/")[-1], include_ei=True)
    eis_one = [with_ttl(vstim_data_one.get_ei_for_cell(int(cell_id)).ei.T).reshape((1, -1, 520)) for cell_id in cell_ids_one]
    eis_two = [with_ttl(vstim_data_two.get_ei_for_cell(int(cell_id)).ei.T).reshape((1, -1, 520)) for cell_id in cell_ids_two]
    eis_one = np.concatenate(eis_one, axis=0)
    eis_two = np.concatenate(eis_two, axis=0)
    print(f"EI tensor dims are {eis_one.shape} and {eis_two.shape}")
    assert eis_one.ndim == 3, "EIS one must be 3D"
    assert eis_two.ndim == 3, "EIS two must be 3D"
    assert eis_one.shape[0] == len(cell_ids_one), "EIS one must have the same number of cells as cell ids one"
    assert eis_two.shape[0] == len(cell_ids_two), "EIS two must have the same number of cells as cell ids two"
    correlation_matrix = correlate_ei_tensors(eis_one, eis_two)

    return correlation_matrix

file_list = os.listdir(os.path.join(ESTIM_ANALYSIS_BASE, DATASET, ESTIM_DATARUN))
elecResps = np.array([f for f in file_list if f.startswith('elecResp_n') and f.endswith('.mat')])

vstims = []
elecResps_new = []
for elecResp in elecResps:
    try: 
        elecResp_data = loadmat(os.path.join(ESTIM_ANALYSIS_BASE, DATASET, ESTIM_DATARUN, elecResp), squeeze_me=True, struct_as_record=False)
        vstim_datarun = elecResp_data['elecResp'].names.rrs_ei_path.split(f'{DATASET}/')[1]
        if len(vstim_datarun.split('/')[0]) > 0:
            if vstim_datarun.split('/')[0].startswith('data'):
                vstim = vstim_datarun.split('/')[0]
            else:
                vstim = "/".join(vstim_datarun.split('/')[:-1])
        else:
            if vstim_datarun.split('/')[1].startswith('data'):
                vstim = vstim_datarun.split('/')[1]
            else:
                vstim = "/".join(vstim_datarun.split('/')[1:-1])
        vstims.append(vstim)
        elecResps_new.append(elecResp)
    except:
        print(f'Error with {elecResp}')
        pass

vstims = np.array(vstims)
elecResps = np.array(elecResps_new)
unique_vstims, unique_idx = np.unique(vstims, return_inverse=True)
print(unique_vstims)

if not os.path.exists(os.path.join(write_path, DATASET, ESTIM_DATARUN)):
    os.makedirs(os.path.join(write_path, DATASET, ESTIM_DATARUN))

for i in range(len(unique_vstims)):
    
    vstim = unique_vstims[i]
    relevant_elecResps = elecResps[np.where(unique_idx == i)[0]]

    if vstim == VSTIM_DATARUN_2:
        for elecResp in relevant_elecResps:
            shutil.copy(os.path.join(ESTIM_ANALYSIS_BASE, DATASET, ESTIM_DATARUN, elecResp),
                        os.path.join(write_path, DATASET, ESTIM_DATARUN, elecResp))
        continue
            
    datapath = os.path.join(VISUAL_ANALYSIS_BASE, DATASET, vstim)
    datarun = vstim.split("/")[-1]
    vcd = vl.load_vision_data(datapath, datarun,
                            include_neurons=True,
                            include_ei=True,
                            include_params=True,
                            include_noise=True)

    cells_1 = np.array(sorted(vcd.get_cell_ids()))

    datapath = os.path.join(VISUAL_ANALYSIS_BASE, DATASET, VSTIM_DATARUN_2)
    datarun = VSTIM_DATARUN_2.split("/")[-1]
    vcd = vl.load_vision_data(datapath, datarun,
                            include_neurons=True,
                            include_ei=True,
                            include_params=True,
                            include_noise=True)
    vcd.update_cell_type_classifications_from_text_file(os.path.join(VISUAL_ANALYSIS_BASE, DATASET, VSTIM_DATARUN_2, f'{datarun}.classification_agogliet.txt'))
    cells_2 = np.array(sorted(vcd.get_cell_ids()))
    cells_2 = np.array([cell_id for cell_id in cells_2 if 'bad' not in vcd.get_cell_type_for_cell(cell_id).lower() and 'dup' not in vcd.get_cell_type_for_cell(cell_id).lower()])

    if len(cells_1) <= len(cells_2):
        # CELLS1 x CELLS2
        corrs = match_eis_across_wn_519(DATASET, vstim, VSTIM_DATARUN_2, cells_1, cells_2, analysis_base=VISUAL_ANALYSIS_BASE)
        flipFlag = 0
    else:
        # CELLS2 x CELLS1
        corrs = match_eis_across_wn_519(DATASET, VSTIM_DATARUN_2, vstim, cells_2, cells_1, analysis_base=VISUAL_ANALYSIS_BASE)
        flipFlag = 1
    
    match_dict = {}
    for elecResp in relevant_elecResps:
        elecResp_data = loadmat(os.path.join(ESTIM_ANALYSIS_BASE, DATASET, ESTIM_DATARUN, elecResp), squeeze_me=True, struct_as_record=False)
        cell_no_1 = elecResp_data['elecResp'].cells.main
        print(f'Running for cell number {cell_no_1}')
        cell_idx = np.where(cells_1 == cell_no_1)[0][0]
        if not(flipFlag):
            match_candidates = np.where(corrs[cell_idx, :] >= CORR_THR)[0]
        else:
            match_candidates = np.where(corrs[:, cell_idx] >= CORR_THR)[0]

        if len(match_candidates) == 0:
            print(f'No match found for cell {cell_no_1}')
        else:
            if not(flipFlag):
                matched_cell = cells_2[match_candidates[np.argmax(corrs[cell_idx, match_candidates])]]
                print(f'Matched cell {cell_no_1} to {matched_cell} with correlation {np.max(corrs[cell_idx, match_candidates])}')
            else:
                matched_cell = cells_2[match_candidates[np.argmax(corrs[match_candidates, cell_idx])]]
                print(f'Matched cell {cell_no_1} to {matched_cell} with correlation {np.max(corrs[match_candidates, cell_idx])}')
                
            
            match_dict[str(cell_no_1)] = matched_cell

            shutil.copy(os.path.join(ESTIM_ANALYSIS_BASE, DATASET, ESTIM_DATARUN, elecResp),
                        os.path.join(write_path, DATASET, ESTIM_DATARUN, 
                                     f'elecResp_n{matched_cell}_{elecResp.split("_")[-2]}_{elecResp.split("_")[-1]}'))

    savemat(os.path.join(write_path, DATASET, ESTIM_DATARUN, 
                         f'match_dict_{DATASET}_{vstim.split("/")[0]}_to_{VSTIM_DATARUN_2.split("/")[0]}.mat'), match_dict)