from mne.stats.cluster_level import (
    _find_clusters, 
    _setup_adjacency,
    _reshape_clusters,  
    _cluster_indices_to_mask
     )
import numpy as np

from .parametric import ARI 
from .permutation import pARI 


def all_resolutions_inference(X, alpha = .05, tail = 0, ari_type = 'parametric',
    adjacency = None, n_permutations = 10000, thresholds = None, 
    seed = None, statfun = None, shift = 0):
    '''
    Implements all-resolutions inference as in [1] or [2].

    Tries a range of cluster thresholds between chosen alpha and the Bonferroni
    corrected threshold. You can manually specify thresholds to try if you want

    Parameters
    ----------
        x: (n_observation, n_times, n_vertices) array for one-sample/paired test
            or a list of two such arrays for independent sample test
        alpha: (float) the false discovry control level
        tail: 1 or 'greater', 0 or 'two-sided', -1 or 'less';
                ignored if statfun is provided.
        adjacency: defines neighbors in the data, as in
                    mne.stats.spatio_temporal_cluster_1samp_test
        type: (str) 'parametric' to perform ARI as in [1] 
                    and 'permutation' as in [2]
        n_permutations: (int) number of permutations to perform
        thresholds: (iterable) optional, manually specify cluster 
                    inclusion thresholds to search over
        shift: (float) shift for candidate critical vector family. 
                Corresponds to delta parameter in [2].
                If statfun p-values are anti-conservative, increasing this can
                increase power for detecting larger clusters (at the cost of 
                decreased power for smaller clusters). Therefore, this
                corresponds to the minimum size cluster we're interested in 
                detecting. Default is 0 (interested in any sized cluster).
                Only for permutation-based ARI, ignored otherwise.
        statfun: a custom statistics function to compute p-values. Should take
                an (n_observations, n_tests) array (or list of such arrays) 
                as input and return an (n_tests,) array of p-values. If this
                argument is used, the tail argument is ignored.

    Returns
    ----------
        p_vals: (sample_shape array) the p-values from the mass univariate test
        true_positive_proportions:  (sample_shape array)
                                    highest true positive proportion
                                    for each coordinate in p_vals
                                    across all thresholds.
        clusters: list of sample_shape boolean masks or empty list
                clusters in which true positive proportion exceeds 1 - alpha

    References
    ----------
    [1] Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ.
        All-Resolutions Inference for brain imaging.
        Neuroimage. 2018 Nov 1;181:786-796.
        doi: 10.1016/j.neuroimage.2018.07.060
    [2] Andreella, Angela, et al. 
        "Permutation-based true discovery proportions for fMRI cluster analysis." 
        arXiv preprint arXiv:2012.00368 (2020).
    '''

    # initialize ARI object, which computes p-value 
    if ari_type == 'parametric':
        ari = ARI(X, alpha, tail, n_permutations, seed, statfun)
    elif ari_type == 'permutation':
        ari = pARI(X, alpha, tail, n_permutations, seed, statfun, shift)
    else:
        raise ValueError("type must be 'parametric' or 'permutation'.")
    p_vals = ari.p_values

    true_discovery_proportions = np.zeros_like(p_vals)
    n_times = p_vals.shape[0]
    n_tests = p_vals.size

    # setup adjacency structure if needed
    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    # handle threshold arguments, construct default if needed
    if thresholds is None: # search grid up to max p-val
        thresholds = np.geomspace(alpha, np.min(p_vals), num = 1000)
    elif thresholds == 'all':
        thresholds = p_vals.flatten()
    else: # verify user-input thresholds 
        if not hasattr(thresholds, '__iter__'):
            thresholds = [thresholds]
        for thres in thresholds:
            # make sure cluster thresholds are valid p-values
            assert(thres >= 0)
            assert(thres <= 1)

    for thres in thresholds:
        if adjacency is None: # use lattice adjacency 
            clusters, _ = _find_clusters(p_vals, thres, -1)
        else:
            clusters, _ = _find_clusters(p_vals.flatten(), thres, -1, adjacency)
        if clusters: # reshape to boolean mask 
            clusters = _cluster_indices_to_mask(clusters, n_tests)
            clusters = _reshape_clusters(clusters, true_discovery_proportions.shape)
        for clust in clusters:
            # compute the true-positive proportion for this cluster
            tdp = ari.true_discovery_proportion(clust)
            # update results array if new TPF > old TPF
            tdp_old = true_discovery_proportions[clust]
            tdp_new = np.full_like(tdp_old, tdp)
            tdps = np.stack([tdp_old, tdp_new], axis = 0)
            true_discovery_proportions[clust] = tdps.max(axis = 0)

    # get clusters where true discovery proportion exceeds threshold
    clusters, _ = _find_clusters(true_discovery_proportions.flatten(), 1 - alpha, 1, adjacency)
    if clusters:
        clusters = _cluster_indices_to_mask(clusters, n_tests)
        clusters = _reshape_clusters(clusters, true_discovery_proportions.shape)
    return p_vals, true_discovery_proportions, clusters
