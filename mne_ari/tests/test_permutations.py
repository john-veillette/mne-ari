from numpy.testing import assert_allclose
from ..permutation import permutation_test
import numpy as np

N_PERM = 500

def _test_one_sample(sample_shape, n_obs = 50):
	'''
    note that even when setting the seed, the permutation p-vals will never be
    exactly equal to their opposite tailed p-vals because of how they're
    computed; each normalizes by (n_perm - 1) instead of by n, and each
    includes the == edge case. Hence, we use assert_allclose.
    '''
	# check that dimensions are handled correctly 
	data = np.random.normal(size = [n_obs] + sample_shape)
	p_vals_pos = permutation_test(data, tail = 1, 
		n_permutations = N_PERM, seed = 0)
	assert(p_vals_pos.shape == tuple(sample_shape))
	p_vals_neg = permutation_test(data, tail = -1, 
		n_permutations = N_PERM, seed = 0)
	assert(p_vals_neg.shape == tuple(sample_shape))
	p_vals_twosided = permutation_test(data, tail = 0, 
		n_permutations = N_PERM, seed = 0)
	assert(p_vals_twosided.shape == tuple(sample_shape))

	# make sure p-values make sense
	assert_allclose(p_vals_pos, 1 - p_vals_neg, atol = 1e-2) 
	p = 2 * np.stack([p_vals_pos, p_vals_neg], axis = 0).min(axis = 0)
	assert_allclose(p_vals_twosided, p, atol = 1e-2)

def _test_two_sample(sample_shape, n_obs1 = 40, n_obs2 = 60):

	# make sure handles dimensions correctly 
	data1 = np.random.normal(size = [n_obs1] + sample_shape)
	data2 = np.random.normal(size = [n_obs2] + sample_shape)
	p_vals_pos = permutation_test([data1, data2], tail = 1, 
		n_permutations = N_PERM, seed = 0)
	assert(p_vals_pos.shape == tuple(sample_shape))
	p_vals_neg = permutation_test([data1, data2], tail = -1, 
		n_permutations = N_PERM, seed = 0)
	assert(p_vals_neg.shape == tuple(sample_shape))
	p_vals_twosided = permutation_test([data1, data2], tail = 0, 
		n_permutations = N_PERM, seed = 0)
	assert(p_vals_twosided.shape == tuple(sample_shape))

	# make sure p-values make sense
	assert_allclose(p_vals_pos, 1 - p_vals_neg, atol = 1e-2) 
	p = 2 * np.stack([p_vals_pos, p_vals_neg], axis = 0).min(axis = 0)
	assert_allclose(p_vals_twosided, p, atol = 1e-2)

def test_permutation_test():
	'''
	make sure permutation test API behaves as anticipated
	'''
	np.random.seed(0)
	# one sample test
	for sample_shape in ([30, 40], [20, 70, 10]):
		_test_one_sample(sample_shape)
		_test_two_sample(sample_shape)
	
