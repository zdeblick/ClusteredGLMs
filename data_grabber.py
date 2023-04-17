import os
import numpy as np
import pandas as pd
from allensdk.core.cell_types_cache import CellTypesCache


def get_sweeps_with_stim(ephys_data,stim_type):
    """
    Input: ephys_data - an object containing all data for a given neuron
             stim_type - a string specifying which stimulus type we want to pull
    Returns: a list of dicts, each containing stimululus and spike time data for an individual trial,
             for all trials of type stim_type contained in ephys_data
    """

    sweeps = []
    for i in ephys_data.get_experiment_sweep_numbers():
        if str.startswith(ephys_data.get_sweep_metadata(i)['aibs_stimulus_name'].decode('ascii'),stim_type):
            sweep = ephys_data.get_sweep(i)
            sweep['spike_times'] = ephys_data.get_spike_times(i)
            sweeps.append(sweep)
    return sweeps

def bin_spikes_and_stim(sweeps,bin_len,trim_0=True):
    """
    Inputs: sweeps - a list of dicts, each containing stimululus and spike time data for an individual trial
            bin_len - desired length of time bin in seconds
            trim_0 - whether to include time bins when there is no stimulus
    Returns: spikes_list - a list of np arrays, each containing the spiking response for a single trial
             stim_list - a list of np arrays, each containing the stimulus (in pA) for a single trial
    """
    spikes_list = []
    stim_list = []
    for s_i, sweep in enumerate(sweeps):
        T = int(np.floor(sweep['index_range'][1]*1.0/(sweep['sampling_rate']*bin_len)))
        binned_spikes = np.zeros((1,T))
        binned_stim = np.empty((1,T))
        for spike_time in sweep['spike_times']:
            t = int(np.floor(spike_time/bin_len))
            binned_spikes[0,t]+=1
        for i in range(T):
            binned_stim[0,i] = 1e9*np.mean(sweep['stimulus'][int(np.round(i*bin_len*sweep['sampling_rate'])):int(np.round((i+1)*bin_len*sweep['sampling_rate']))])
        if trim_0:
            binned_spikes = binned_spikes[np.abs(binned_stim)>1e-3]
            binned_stim = binned_stim[np.abs(binned_stim)>1e-3]
        spikes_list.append(binned_spikes)
        stim_list.append(binned_stim)
    return spikes_list, stim_list

ctc = CellTypesCache()
cells = pd.DataFrame(ctc.get_cells(reporter_status='positive'))
print(len(cells[(cells['reporter_status']=='positive') & (cells['structure_area_abbrev']=='VISp')]))
all_new_cells = cells[cells['structure_area_abbrev']=='VISp']

cell_ids = all_new_cells['id'].values

bin_len = 0.002 #2ms

# Train data, "Noise 1"
bspks = []
bstms = []
# Test trials, "Noise 2"
test_bspks = []
test_bstms = []
print(len(cell_ids))
# Neurons excluded for lack of data
drop_inds = []

for num in range(len(cell_ids)):
    print(num)
    train_sweeps = get_sweeps_with_stim(ctc.get_ephys_data(cell_ids[num]),'Noise 1')
    test_sweeps = get_sweeps_with_stim(ctc.get_ephys_data(cell_ids[num]),'Noise 2')
    if len(train_sweeps)>2 and len(test_sweeps)>2:
        (binned_spikes, binned_stim) = bin_spikes_and_stim(train_sweeps,bin_len)
        bspks.append(binned_spikes)
        bstms.append(binned_stim)
        (binned_spikes, binned_stim) = bin_spikes_and_stim(test_sweeps,bin_len)
        test_bspks.append(binned_spikes)
        test_bstms.append(binned_stim)
    else:
        print(len(train_sweeps), len(test_sweeps))
        drop_inds.append(num)
        print('drop')
    
test_spk_counts = [np.sum([np.sum(s) for s in test_bspks[n]]) for n in range(len(test_bspks))]

# dataframe with metadata for included cells
all_new_cells = all_new_cells.drop(all_new_cells.index[drop_inds])
all_new_cells.to_csv('n12_cells.csv')

# binned_spikes - list of lists of np arrays (neurons by trials by time bins) of spiking responses to "Noise 1"
# binned_stim - list of lists of np arrays (neurons by trials by time bins) of binned stimulus to "Noise 1"
# test_binned_spikes, test_binned_stim - same as above, for "Noise 2"
# bin_len - length of time bins in seconds
# test_spk_counts - list (neurons), the total number of spikes each neuron fires in response to all "Noise 2" trials
np.savez('ivscc_data_n12',binned_spikes=bspks,binned_stim=bstms,
    test_binned_spikes=test_bspks,test_binned_stim=test_bstms,
    bin_len=bin_len,test_spk_counts=test_spk_counts)

