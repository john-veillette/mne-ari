from mne.utils import check_random_state
from typing import Iterable
import numpy as np 

'''
This module provides mass univariate permutation tests for mutlidimensional 
arrays. Note that the (univariate, not cluster-based) permutation test
provided in MNE as mne.stats.permutation_t_test applies a t-max correction
for multiple comparisons to its p-values, making those p-values unsuitable for
all-resolutions inference. Hence, we provide our own implementations here. 

They're not the fastest permutation tests in the world with those for loops,
but they're pretty memory efficient since they never hold the full permutation
distribution in memory at once.
'''

def _compare(obs, perm, tail):
    if tail == 1:
        mask = (perm >= obs)
    elif tail == -1:
        mask = (perm <= obs)
    return mask

def _permutation_test_1samp(X, n_permutations = 10000, tail = 0, seed = None):
    rng = check_random_state(seed)
    greater_ct = np.zeros_like(X[0])
    lesser_ct = np.zeros_like(X[0])
    obs = X.mean(0)
    for i in range(n_permutations):
        flips = rng.choice([-1, 1], size = X.shape[0])
        broadcast_shape = [X.shape[0]] + (len(X.shape) - 1)*[1]
        flips = np.reshape(flips, broadcast_shape)
        perm_X = flips * X
        perm_effect = perm_X.mean(0)
        greater_ct += _compare(obs, perm_effect, 1)
        lesser_ct += _compare(obs, perm_effect, -1)
    if tail == 1:
        p = (greater_ct + 1) / (n_permutations + 1)
    elif tail == -1:
        p = (lesser_ct + 1) / (n_permutations + 1)
    elif tail == 0:
        p1 = (greater_ct + 1) / (n_permutations + 1)
        p2 = (lesser_ct + 1) / (n_permutations + 1)
        p = 2 * np.stack([p1, p2], 0).min(0)
    else:
        raise ValueError("Cannot compute p-value with meaningless tail = %d."%tail)
    return p

def _permutation_test_ind(X, n_permutations = 10000, tail = 0, seed = None):
    rng = check_random_state(seed)
    n1 = len(X[0])
    X = np.concatenate(X, axis = 0)
    greater_ct = np.zeros_like(X[0])
    lesser_ct = np.zeros_like(X[0])
    idxs = np.arange(len(X))
    obs = X[:n1].mean(0) - X[n1:].mean(0)
    for i in range(n_permutations):
        rng.shuffle(idxs)
        perm_X = X[idxs]
        perm_effect = perm_X[:n1].mean(0) - perm_X[n1:].mean(0)
        greater_ct += _compare(obs, perm_effect, 1)
        lesser_ct += _compare(obs, perm_effect, -1)
    if tail == 1:
        p = (greater_ct + 1) / (n_permutations + 1)
    elif tail == -1:
        p = (lesser_ct + 1) / (n_permutations + 1)
    elif tail == 0:
        p1 = (greater_ct + 1) / (n_permutations + 1)
        p2 = (lesser_ct + 1) / (n_permutations + 1)
        p = 2 * np.stack([p1, p2], 0).min(0)
    else:
        raise ValueError("Cannot compute p-value with meaningless tail = %d."%tail)
    return p


def permutation_test(X, **kwargs):
    """
    Parameters
    ----------
    X : array, shape (n_samples, n_tests) if one-sample;
        list of 2 arrays if independent samples 
        Samples (observations) by number of tests (variables).
    n_permutations : int, Number of permutations.
    tail : -1 or 0 or 1 (default = 0)
        If tail is 1, the alternative hypothesis is that the
        mean of the data is greater than 0 (upper tailed test).  If tail is 0,
        the alternative hypothesis is that the mean of the data is different
        than 0 (two tailed test).  If tail is -1, the alternative hypothesis
        is that the mean of the data is less than 0 (lower tailed test).
    """
    if isinstance(X, list) or isinstance(X, tuple):
        assert(len(X) == 2)
        assert(X[0].shape[1:] == X[1].shape[1:])
        return _permutation_test_ind(X, **kwargs)
    else:
        return _permutation_test_1samp(X, **kwargs)


