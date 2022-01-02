from ._permutation import _permutation_1samp, _permutation_ind
import numpy as np

def _optimize_lambda(p, alpha, delta = 0):
    '''
    finds best lambda parameter given the permutation distribution of p-values

    based on https://github.com/angeella/pARI/blob/master/src/lambdaCalibrate.cpp
    but only supports Simes family 
    '''
    b = p.shape[1] # number of permutations 
    mm = p.shape[0] # number of tests
    T = np.empty(b)
    idV = np.arange(1 + delta, 1 + mm)
    deltaV = np.full(mm - delta, delta)
    mV = np.full(mm - delta, mm)

    ## sort columns of p-vals
    Y = np.sort(p, axis = 1)

    # compute lambda for each permutation
    for bb in range(b):
        lam = (mV - deltaV) % Y[delta:mm, bb] / ((idV - deltaV) * alpha)
        T[bb] = np.min(lam) # minimum over hypotheses
    T = np.sort(T)
    idx = np.floor(alpha * b).astype(int)
    return T[idx]

def _get_critical_vector(p, alpha, lam, delta = 0):
    '''
    based on https://github.com/angeella/pARI/blob/master/R/criticalVector.R
    but only support Simes family 
    '''
    m = p.shape[0]
    cc = np.arange(1, m + 1)
    vf = np.vectorize(lambda x: ((x - delta) * alpha * lam) / (m - delta))
    return vf(cc)


class pARI:
    '''
    A class that handles permuation-based All-Resolutions Inference as in [1].

    [1] Andreella, Angela, et al. 
        "Permutation-based true discovery proportions for fMRI cluster analysis." 
        arXiv preprint arXiv:2012.00368 (2020).
    '''

    def __init__(self, X, alpha, tail = 0, 
        n_permutations = 10000, seed = None, statfun = None, shift = 0):
        '''
        uses permutation distribution to estimate best critical vector
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
        assert(shift >= 0)
        self.delta = shift # same default as pARIBrain, see [1] 

        if type(X) in [list, tuple]:
            self.sample_shape = X[0][0].shape 
            X = [np.reshape(x, (x.shape[0], -1)) for x in X] # flatten samples
            p = _permutation_ind(X, n_permutations, self.alternative, seed, statfun)
        else:
            self.sample_shape = X[0].shape
            X = np.reshape(X, (X.shape[0], -1)) # flatten samples
            p = _permutation_1samp(X, n_permutations, self.alternative, seed, statfun)

        self.p = p[:, 0] # just the observed values 
        self.lam = _optimize_lambda(p, self.alpha, self.delta)
        self.crit_vec = _get_critical_vector(p, self.alpha, self.lam, self.delta)

    def true_discovery_proportion(self, mask):
        '''
        given a boolean mask, gives the true discovery proportion
        for the specified cluster 
        '''
        assert(mask.shape == self.sample_shape)
        mask = mask.flatten()
        m = mask.sum() # number of tests in mask 
        p = self.p # observed p-values 
        p_vec = p[mask] # p-values in subset 
        u = np.empty(m)
        for i in range(m):
            u[i] = np.sum(p_vec <= self.crit_vec[i]) - i 
        n_discoveries = np.max(u) # a lower bound
        tdp = n_discoveries / m
        try:
            assert(tdp >= 0)
            assert(tdp <= 1)
        except:
            raise Exception("Something weird happened," +
                " and we got a TDP outside of the range [0, 1]." +
                " Did you use a custom stat function?" +
                " Are you sure your p-values make sense?")
        return tdp

    @property 
    def p_values(self):
        return np.reshape(self.p, self.sample_shape)


