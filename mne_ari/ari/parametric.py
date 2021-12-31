from ._permutation import _permutation_1samp, _permutation_ind
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
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_samples, 1, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(n_samples + (alpha - slope * n_samples) / slope)
    return np.minimum(hommel_value, n_samples)

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

class ARI:
    '''
    A class that handles parametric All-Resolutions Inference as in [1].

    [1] Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ.
        All-Resolutions Inference for brain imaging.
        Neuroimage. 2018 Nov 1;181:786-796.
        doi: 10.1016/j.neuroimage.2018.07.060
    '''

    def __init__(self, X, alpha, tail = 0, n_permutations = 10000, seed = None):
        '''
        use permutation distribution to estimate best critical vector for later inference
        '''
        if tail == 0 or tail == 'two-sided':
            self.alternative = 'two-sided'
        elif tail == 1 or tail == 'greater':
            self.alternative = 'greater'
        elif tail == -1 or tail == 'less':
            self.alternative = 'less'
        else:
            raise ValueError('Invalid input value for tail!')
        self.alpha = alpha

        if type(X) in [list, tuple]:
            self.sample_shape = X[0][0].shape 
            X = [np.reshape(x, (x.shape[0], -1)) for x in X] # flatten samples
            p = _permutation_ind(X, n_permutations, self.alternative, seed)
        else:
            self.sample_shape = X[0].shape
            X = np.reshape(X, (X.shape[0], -1)) # flatten samples
            p = _permutation_1samp(X, n_permutations, self.alternative, seed)

        p_obs = p[:, 0] # observed p-values from t-test
        self.p = (p_obs[:, np.newaxis] <= p).mean(1) # permutation p-values
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
        return tdp

    @property 
    def p_values(self):
        return np.reshape(self.p, self.sample_shape)