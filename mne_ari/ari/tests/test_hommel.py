from ..parametric import _compute_hommel_value, _true_positive_fraction
from nilearn.glm.thresholding import _compute_hommel_value as nl_compute_hommel_value
from nilearn.glm.thresholding import _true_positive_fraction as nl_true_positive_fraction

from numpy.testing import assert_allclose
from scipy.stats import norm
import numpy as np

def _test_hommel(alpha = .05):
    '''
    verify that results are equivalent between this and nilearn's
    implementation of parametric ARI.
    '''
    p_vals = np.random.uniform(size = 100)
    hom = _compute_hommel_value(p_vals, alpha)
    z_vals = norm.isf(p_vals)
    hom_nl = nl_compute_hommel_value(z_vals, alpha)
    assert(hom == hom_nl)
    tpf = _true_positive_fraction(p_vals, hom, alpha)
    tpf_nl = nl_true_positive_fraction(z_vals, hom, alpha)
    assert_allclose(tpf, tpf_nl, atol = 1e-5)

def test_hommel():
    np.random.seed(0)
    for i in range(100):
        _test_hommel()

