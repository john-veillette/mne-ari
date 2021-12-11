from mne.stats.cluster_level import _find_clusters, _setup_adjacency
import numpy as np

def _compute_hommel_value(p_vals, alpha):
    '''
    compute all-resolutions inference Hommel value

    **implementation modified from nilearn.glm.thresholding**
    '''
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
    hommel_value = _compute_hommel_value(p_vals, alpha)
    n_samples = len(p_vals)
    c = np.ceil((hommel_value * p_vals) / alpha)
    unique_c, counts = np.unique(c, return_counts = True)
    criterion = 1 - unique_c + np.cumsum(counts)
    proportion_true_discoveries = np.maximum(0, criterion.max() / n_samples)
    return proportion_true_discoveries

def all_resolutions_inference(p_vals, threshold = .05, alpha = .05, adjacency):
    '''
    Implements all-resolutions inference as in [1].

    Parameters:
        p_vals: (n_times, n_vertices) array
        threshold: (float or iterable) p-value threshold(s) for
                    inclusion in a cluster.
        alpha: (float) a "true-discovery" should still have p-value < alpha
                    after accounting for multiple comparisons. So this is
                    the false discovry control level.
        adjacency: defines neighbors in the data, as in
                    mne.stats.spatio_temporal_cluster_1samp_test

    Returns:
        true_positive_proportions:  highest true positive proportion
                                    for each coordinate in p_vals
                                    across all input thresholds.

    [1] Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ.
    All-Resolutions Inference for brain imaging.
    Neuroimage. 2018 Nov 1;181:786-796.
    doi: 10.1016/j.neuroimage.2018.07.060
    '''
    n_times = p_vals.shape[1]
    n_elecs = p_vals.shape[2]
    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)
    hom = _compute_hommel_value(p_vals, alpha)
    true_positive_proportions = np.zeros_like(p_vals)
    for thres in threshold:
        clusters, _ = _find_clusters(p_vals, thres, -1, adjacency)
        for clust in clusters:
            # compute the true-positive proportion for this cluster
            clust_ps = p_vals[clust]
            tpf = _true_positive_fraction(clust_ps, hom, alpha)
            # update results array if new TPF > old TPF
            tpf_max = true_positive_proportion[clust]
            tpfs = np.stack([tpf_max, tpf], axis = 0)
            tpf_max = tpfs.max(axis = 0)
            true_positive_proportion[clust] = tpf_max
    return true_positive_proportions
