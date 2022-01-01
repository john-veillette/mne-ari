from scipy.stats import ttest_1samp, ttest_ind
from mne.utils import check_random_state
import numpy as np 

def _permutation_1samp(X, n_permutations = 10000, alternative = 'two-sided', 
	seed = None, statfun = None):
    '''
    computes the permutation distribution of p-values from ttest_1samp
    '''
    _, p_obs = ttest_1samp(X, 0, axis = 0, alternative = alternative)
    p_dist = [p_obs]

    rng = check_random_state(seed)
    for i in range(n_permutations):
        # randomly flip sign of observations 
        flips = rng.choice([-1, 1], size = X.shape[0])
        X = X * flips[:, np.newaxis] 
        # and recompute test statistic
        if statfun is None:
        	_, p_obs = ttest_1samp(X, 0, axis = 0, alternative = alternative)
        else:
        	p = statfun(X)
        p_dist.append(p_obs)
    return np.stack(p_dist, axis = 1) # n_tests x (1 + n_perm)


def _permutation_ind(X, n_permutations = 10000, alternative = 'two-sided', 
	seed = None, statfun = None):
    '''
    permutation distribution of parametric p-values from ttest_ind
    '''
    if len(X) != 2 and statfun is None:
    	raise ValueError("You're trying do do a two-sample test " + 
    		"with a number of samples that isn't two! If X is list/tuple, " +
    		"it must be of length 2.")
    _, p_obs = ttest_ind(X[0], X[1], axis = 0, alternative = alternative)
    p_dist = [p_obs]

    n = X[0].shape[0] # number of observations just for first sample
    X = np.concatenate(X, axis = 0)
    idxs = np.arange(X.shape[0])
    rng = check_random_state(seed)
    for i in range(n_permutations):
        rng.shuffle(idxs)
        perm_X = X[idxs]
        X0 = perm_X[:n]
        X1 = perm_X[n:]
        if statfun is None:
        	_, p = ttest_ind(X0, X1, axis = 0, alternative = alternative)
        else:
        	p = statfun([X0, X1])
        p_dist.append(p)

    return np.stack(p_dist, axis = 1) # n_tests x (1 + n_perm)