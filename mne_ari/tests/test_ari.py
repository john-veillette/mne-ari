from mne.datasets.sample import data_path
from mne.channels import find_ch_adjacency
from ..ari import all_resolutions_inference
import numpy as np
from os.path import join  

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
	tdp, clusters = all_resolutions_inference(ari, adjacency = adjacency)
	assert(tdp.shape == p_vals.shape)
	assert (tdp <= 1)
	assert(tdp >= 0)
	assert(type(clusters) is list)
	if len(clusters) > 0:
		for i in range(len(clusters)):
			assert(clusters[i].shape == p_vals.shape)

def test_ari():
	np.random.seed(0)
	# get an example adjacency matrix 
	fpath = join(data_path(), 'MEG', 'sample', 'sample_audvis_filt-0-40_raw.fif')
	raw = mne.io.read_raw_fif(fpath, preload = False)
	adjacency, ch_names = find_ch_adjacency(raw.info, ch_type = 'mag')

	# test various sample shapes...
	# last dimension should be dimension adjacency matrix applies to, 
	# so must be same length as ch_names.
	n_chan = len(ch_names)
	for i in range(5): # ARI should handle an arbitrary number of dimensions
		shape = np.random.randint(10, 200, size = i).tolist() + [n_chan]
		# test with and without adjacency 
		_test_ari(shape)
		_test_ari(shape, adjacency)




