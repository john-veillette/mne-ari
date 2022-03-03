from ._permutation import _permutation_1samp, _permutation_ind
from ..permutation import permutation_test
import numpy as np

def _compute_hommel_value(p_vals, alpha):
    '''
    compute all-resolutions inference Hommel value

    **implementation modified from nilearn.glm.thresholding**
    '''
    p_vals = p_vals.copy().flatten()
    p_vals = np.sort(p_vals)
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    n_samples = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_samples
    if p_vals[-1] < alpha:
        return 0
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_samples - 1, 0, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(alpha / slope)
    return int(np.minimum(hommel_value, n_samples))

def _true_positive_fraction(p_vals, hommel_value, alpha):
    '''
    Given a bunch of p-values, return the true positive fraction

    **implementation modified from nilearn.glm.thresholding**

    Parameters
    ----------
    p_vals : array,
        A set of p-values from which the TPF is computed.
    hommel_value: int
        The Hommel value, used in the computations.
    alpha : float
        The desired FDR control.

    Returns
    -------
    proportion_true_discoveries : float
        Estimated true positive fraction in the set of values.
    '''
    p_vals = p_vals.copy().flatten()
    p_vals = np.sort(p_vals)
    n_samples = len(p_vals)
    c = np.ceil((hommel_value * p_vals) / alpha)
    unique_c, counts = np.unique(c, return_counts = True)
    criterion = 1 - unique_c + np.cumsum(counts)
    proportion_true_discoveries = np.maximum(0, criterion.max() / n_samples)
    return proportion_true_discoveries

def statfun_warning():
    '''
    raised when user inputs a custom statistics function
    '''
    import warnings
    warnings.warn(
'''
Parametric ARI assumes that the p-values of individual tests are valid when
calculating the true positive proprtion. If this isn't likely true (e.g.
you're using a parametric test on M/EEG data), you may want to consider using
permutation-based ARI instead.
'''
    )

class ARI:
    '''
    A class that handles parametric All-Resolutions Inference as in [1].

    [1] Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ.
        All-Resolutions Inference for brain imaging.
        Neuroimage. 2018 Nov 1;181:786-796.
        doi: 10.1016/j.neuroimage.2018.07.060
    '''

    def __init__(self, X, alpha, tail = 0,
        n_permutations = 10000, seed = None, statfun = None):
        '''
        use permutation distribution to estimate best critical vector for later inference
        '''
        if tail == 0 or tail == 'two-sided':
            self.alternative = 0
        elif tail == 1 or tail == 'greater':
            self.alternative = 1
        elif tail == -1 or tail == 'less':
            self.alternative = -1
        else:
            raise ValueError('Invalid input value for tail!')
        self.alpha = alpha

        if type(X) in [list, tuple]:
            self.sample_shape = X[0][0].shape
            X = [np.reshape(x, (x.shape[0], -1)) for x in X] # flatten samples
            if statfun is None:
                # we use our own permutation test rather than e.g. scipy's b/c
                # those are slower and can give incorrect p-vals == 0 exactly
                # if observed as greater/less than all random shuffles
                try:
                    assert(len(X) == 2)
                except:
                    raise ValueError("X list must be of length 2 " +
                        " for default independent sample statfun")
                p = permutation_test(
                    X,
                    n_permutations = n_permutations, tail = self.alternative,
                    seed = seed
                    )
            else:
                statfun_warning()
                p = statfun(X)
        else:
            self.sample_shape = X[0].shape
            X = np.reshape(X, (X.shape[0], -1)) # flatten samples
            if statfun is None:
                p = permutation_test(
                    X,
                    n_permutations = n_permutations, tail = self.alternative,
                    seed = seed
                    )
            else:
                statfun_warning()
                p = statfun(X)

        # ARI can output TDP > 1 if p == 0, neither of which make sense
        if np.any(p == 0):
            raise ValueError(">= 1 of your p-values is exactly zero." +
                    " This isn't valid and will cause ARI to behave poorly." +
                    " Try changing your stat function.")
        self.p = p
        self.hommel = _compute_hommel_value(self.p, self.alpha)

    def true_discovery_proportion(self, mask):
        '''
        given a boolean mask of sample_shape, gives the true discovery proportion
        for the specified cluster
        '''
        assert(mask.shape == self.sample_shape)
        mask = mask.flatten()
        p = self.p # observed p-values
        p = p[mask] # only p-vals in cluster
        tdp = _true_positive_fraction(p, self.hommel, self.alpha)
        try:
            assert(tdp >= 0)
            assert(tdp <= 1)
        except:
            raise Exception("Something weird happened," +
                " and we got a TDP outside of the range [0, 1]." +
                " Did you use a custom stat function?" +
                " Are you sure you p-values make sense?")
        return tdp

    @property
    def p_values(self):
        return np.reshape(self.p, self.sample_shape)
