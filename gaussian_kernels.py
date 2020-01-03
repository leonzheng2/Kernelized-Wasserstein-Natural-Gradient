"""
Implementation of Gaussian kernels.
Leon Zheng
"""

import numpy as np

def gaussian_kernel(a, b, sigma):
    """
    Gaussian kernel
    :param a: vector (d, )
    :param b: vector (d, )
    :param sigma: float
    :return: float
    """
    return np.exp(- np.linalg.norm(a - b)**2 / (2*sigma**2))

def grad_gaussian_kernel(k, l, a, b, sigma):
    """
    Compute \partial_k \partial_{l+d} k(a, b), for k, l \in [1,d]
    :param k: int in [1,d]
    :param l: int in [1,d]
    :param a: vector (d, )
    :param b: vector (d, )
    :param sigma: float
    :return: float
    """
    return 1/sigma**2 * gaussian_kernel(a, b, sigma) * (-1/sigma**2 * (a[k] - b[k]) * (a[l] - b[l]) + (l == k))
