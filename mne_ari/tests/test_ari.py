from mne.datasets.sample import data_path
from mne.channels import find_ch_adjacency
from ..ari import all_resolutions_inference
import numpy as np
from os.path import join  
import mne

def _test_ari(sample_shape, adjacency = None):
	'''
	check that ARI API is behaving as expected for a given sample shape

	There isn't a good reference implementation in Python to compare results to
	(other than the nilearn one that our helper functions in ari.py basically
	copy), so the efficacy of our implementation isn't verified as part of the
	continuous integration pipeline. Instead, we test the false positive 
	characteristics manually by simulation.
	'''
	p_vals = np.random.random(sample_shape)
	tdp, clusters = all_resolutions_inference(p_vals, adjacency = adjacency)
	assert(tdp.shape == p_vals.shape)
	assert (np.sum(tdp <= 1) == tdp.size)
	assert(np.sum(tdp >= 0) == tdp.size)
	assert(type(clusters) is list)
	if len(clusters) > 0:
		for i in range(len(clusters)):
			assert(clusters[i].shape == p_vals.shape)

def test_ari():

	np.random.seed(0)

	# get an example adjacency matrix from MNE test data 
	sample_data_dir = mne.datasets.testing.data_path()
	eeg_fpath = join(sample_data_dir, 'EEGLAB', 'test_raw_onefile.set')
	raw = mne.io.read_raw_eeglab(eeg_fpath, preload = True)
	locs_fpath = join(sample_data_dir, 'EEGLAB', 'test_chans.locs')
	dig = mne.channels.read_custom_montage(locs_fpath)
	mapping = {raw.ch_names[i]: dig.ch_names[i] for i in range(len(raw.ch_names))}
	raw = raw.rename_channels(mapping)
	raw = raw.set_montage(dig)
	adjacency, ch_names = find_ch_adjacency(raw.info, ch_type = 'eeg')

	# test various sample shapes...
	# last dimension should be dimension adjacency matrix applies to, 
	# so must be same length as ch_names.
	n_chan = len(ch_names)
	for i in range(3): # ARI should handle an arbitrary number of dimensions
		shape = np.random.randint(10, 50, size = i).tolist() + [n_chan]
		# test with and without adjacency 
		_test_ari(shape)
		if len(shape) < 3: # higher dimensional arrays require well-specified adjacency as shown in:
		# https://mne.tools/stable/auto_tutorials/stats-sensor-space/40_cluster_1samp_time_freq.html
			_test_ari(shape, adjacency)
		elif len(shape) == 3: # n_dim == 3
			adj = mne.stats.combine_adjacency(adjacency, shape[1], shape[0])
			_test_ari(shape, adj)




