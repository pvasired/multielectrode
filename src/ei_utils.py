def get_collapsed_ei_thr(datapath, datarun, cell_no, thr_factor):
    vcd = vl.load_vision_data(datapath, datarun,
                          include_neurons=True,
                          include_ei=True,
                          include_params=True,
                          include_noise=True)
    # Read the EI for a given cell
    cell_ei = vcd.get_ei_for_cell(cell_no).ei
    
    # Collapse into maximum value
    collapsed_ei = np.amin(cell_ei, axis=1)
    
    channel_noise = vcd.channel_noise
    
    # Threshold the EI to pick out only electrodes with large enough values
    good_inds = np.argwhere(np.abs(collapsed_ei) > thr_factor * channel_noise).flatten()
    
    return good_inds, np.abs(collapsed_ei)
