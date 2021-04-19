# General file for gathering the simulation data from WFSim

import wfsim
import strax
import straxen
import sys
import argparse

from example_instructions import single_electron_instructions
import numpy as np
import pickle as pk

AUX_REPO = 'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'
CONFIG_1T_HASH = '8b3c60a1d790dbcc9103559d0e019008b8e760dc/' # Last commit was June 26, 2020
CONFIG_nT_HASH = '4e71b8a2446af772c83a8600adc77c0c3b7e54d1/' # Last commit was June 24, 2020
GAIN_HASH = '58e615f99a4a6b15e97b12951c510de91ce06045/'      # Last commit was Dec, 2019


# Take the run_id from command line argument (first arg)
parser = argparse.ArgumentParser()

parser.add_argument('-r', '--run_id', type=int, required=True,
        help="run_id that is also used for the random seed")
parser.add_argument('-d', '--detector', type=str, required=True,
        help="choose the detector type: nt or 1t")

# Although an argument could be added for which instructions to use, I choose to leave it out. The
# instruction functions are specific to what is wanting to be simulated. For more complex
# instructions, it should be upon the user to define this instruction and import it into here.

args = parser.parse_args()
detector = args.detector
run_id = str(args.run_id)

np.random.seed( int(run_id) )

if "1t" in detector.lower():
    st = strax.Context(storage=strax.DataDirectory('./strax_data'),
                       register=wfsim.RawRecordsFromFax,
                       config=dict(detector='XENON1T',
                                  fax_config=AUX_REPO+CONFIG_1T_HASH+'fax_files/fax_config_1t.json',
                                  **straxen.contexts.x1t_common_config),
                       timeout=3600,
                       **straxen.contexts.common_opts)
    DETECTOR='1T'
elif "nt" in detector.lower():
    straxen.contexts.xnt_common_config['gain_model'] = ('to_pe_per_run',
            AUX_REPO+GAIN_HASH+'fax_files/to_pe_nt.npy')
    st = strax.Context(storage=strax.DataDirectory('./strax_data'),
                       register=wfsim.RawRecordsFromFax,
                       config=dict(detector='XENONnT',
                                  fax_config=AUX_REPO+CONFIG_nT_HASH+'fax_files/fax_config_nt.json',
                                  to_pe_file=AUX_REPO+GAIN_HASH+'fax_files/to_pe_nt.npy',
                                  **straxen.contexts.xnt_common_config),
                       timeout=3600,
                       **straxen.contexts.common_opts)
    DETECTOR='nT'
else:
    from sys import exit
    exit("Detector option give was not one of the ones available: nT or 1T.")

# Replacing wfsim's built-in rand_instructions
wfsim.strax_interface.rand_instructions = single_electron_instructions # Single electron requires filtering of 0 electron events

st.set_config(dict(fax_files=None, nchunk=100, event_rate=1, chunk_size=500))

# Wanting perfect
st.set_config(dict(field_distortion_on=False))

peaks = st.get_array(run_id, 'peaks')
truth = st.get_array(run_id, 'truth')
basics = st.get_array(run_id, 'peak_basics')

# There will be various truths that are not type 2 (not S2)
good_truth = truth[truth['type'] == 2]
good_truth = good_truth[good_truth['n_electron'] == 1] # Want single electrons only

# Removing the non-S2s gets events out of sync, so I'll match them based on times
good_pk_mask = np.zeros(len(basics), dtype=np.bool)
for i, t in enumerate(good_truth):
    good_pk_mask += ( (basics['time'] < t['t_mean_photon']) &
                      (basics['endtime'] > t['t_mean_photon']) )

peaks = peaks[good_pk_mask]


if len(peaks['time']) != len(good_truth['time']):
    # Making sure that the syncs are proper. Issue mainly shows up for single electron instructions
    true_time = good_truth['time']
    sim_time = peaks['time']
    bad_indices = []
    diff_tol = 1000 # The time difference shouldn't be large. Definitely not larger than this
    max_diff = diff_tol+1
    while max_diff > diff_tol:
        max_diff = 0
        for idx, st in enumerate(sim_time):
            diff = abs(st - true_time[idx])
            if diff > diff_tol:
                bad_indices.append(idx)
                true_time = np.delete(true_time, idx)
                break

hit_pattern_top = peaks['area_per_channel'][:,:253]
event_pos = np.vstack( (good_truth['x'], good_truth['y']) ).T
events = np.zeros(len(event_pos),
                  dtype=[('area_per_channel', np.float32, hit_pattern_top.shape[1]),
                         ('true_pos', np.float32, 2)])

events['area_per_channel'] = hit_pattern_top
events['true_pos'] = event_pos

with open('./pickles/{}-50000_single-el_z-10_sim_{}.pkl'.format(run_id, DETECTOR), 'wb') as fn:
    pk.dump(events, fn)

