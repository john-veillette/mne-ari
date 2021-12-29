from mne.stats.cluster_level import (
    _find_clusters, 
    _setup_adjacency,
    _reshape_clusters,  
    _cluster_indices_to_mask
     )
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
    Given a bunch of z-avalues, return the true positive fraction

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

def all_resolutions_inference(p_vals, alpha = .05, adjacency = None, thresholds = None):
    '''
    Implements all-resolutions inference as in [1].

    Tries a range of cluster thresholds between chosen alpha and the Bonferroni
    corrected threshold, spaced evenly on a log scale.

    Parameters
    ----------
        p_vals: (n_times, n_vertices) array
        alpha: (float) the false discovry control level
        adjacency: defines neighbors in the data, as in
                    mne.stats.spatio_temporal_cluster_1samp_test
        thresholds: (iterable) optional, manually specify cluster 
                    inclusion thresholds to search over

    Returns
    ----------
        true_positive_proportions:  highest true positive proportion
                                    for each coordinate in p_vals
                                    across all thresholds.
        clusters: clusters in which true positive proportion exceeds 1 - alpha

    [1] Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ.
        All-Resolutions Inference for brain imaging.
        Neuroimage. 2018 Nov 1;181:786-796.
        doi: 10.1016/j.neuroimage.2018.07.060
    '''
    true_positive_proportions = np.zeros_like(p_vals)
    n_times = p_vals.shape[0]
    n_tests = p_vals.size

    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    hom = _compute_hommel_value(p_vals, alpha)

    if thresholds is None: # search grid up to Bonferroni corrected alpha
        thresholds = np.linspace(alpha, alpha / p_vals.size, num = 50)
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
        if clusters:
            clusters = _cluster_indices_to_mask(clusters, n_tests)
            clusters = _reshape_clusters(clusters, true_positive_proportions.shape)
        for clust in clusters:
            # compute the true-positive proportion for this cluster
            clust_ps = p_vals[clust]
            tpf = _true_positive_fraction(clust_ps, hom, alpha)
            # update results array if new TPF > old TPF
            tpf_max = true_positive_proportions[clust]
            tpf = np.full_like(tpf_max, tpf)
            tpfs = np.stack([tpf_max, tpf], axis = 0)
            tpf_max = tpfs.max(axis = 0)
            true_positive_proportions[clust] = tpf_max

    # get clusters where true discovery proportion exceeds threshold
    clusters, _ = _find_clusters(true_positive_proportions.flatten(), 1 - alpha, 1, adjacency)
    if clusters:
        clusters = _cluster_indices_to_mask(clusters, n_tests)
        clusters = _reshape_clusters(clusters, true_positive_proportions.shape)
    return true_positive_proportions, clusters
