"""
Implementation of the estimator of Kernelized Wasserstein Natural Gradient, with Gaussian kernel
Leon Zheng
"""

import numpy as np
from gaussian_kernels import grad_gaussian_kernel
from multi_normal_model import h_theta, grad_theta_h_theta, grad_L_theta

"""
Kernelized Wasserstein Natural Gradient
"""

def compute_element_for_inverse_metric(X, Y, idx, gaussian_kernel_sigma):
    """
    Return matrices to compute the estimation of the inverse metric in Wasserstein natural gradient.
    :param X:
    :param Y:
    :param idx:
    :param gaussian_kernel_sigma:
    :return:
    """
    N = X.shape[0]
    d = X.shape[1]
    M = Y.shape[0]
    # Matrix C
    C = np.zeros((M, N * d))
    for m in range(M):
        for n in range(N):
            for i in range(d):
                C[m, n * d + i] = grad_gaussian_kernel(idx[m], i, Y[m], X[n], gaussian_kernel_sigma)
    # Matrix K
    K = np.zeros((M, M))
    for m1 in range(M):
        for m2 in range(M):
            K[m1, m2] = grad_gaussian_kernel(idx[m1], idx[m2], Y[idx[m1]], Y[idx[m2]], gaussian_kernel_sigma)
    # Gradient of kernel for computing T
    grad_kernel_tensor = np.zeros((M, N, d))
    for m in range(M):
        for n in range(N):
            grad_kernel_tensor[m, n] = np.array([grad_gaussian_kernel(idx[m], i, Y[m], X[n], gaussian_kernel_sigma) for i in range(d)])

    return C, K, grad_kernel_tensor


def estimate_wasserstein_inverse_metric(theta, X, Y, Z, lamb, eps, C, K, grad_kernel_tensor):
    """
    Estimate the Wasserstein inverse metric with the samples.
    :param theta:
    :param X:
    :param Y:
    :param Z:
    :param lamb:
    :param eps:
    :param C:
    :param K:
    :param grad_kernel_tensor:
    :return:
    """
    N = X.shape[0]
    M = Y.shape[0]
    q = theta.shape[0] * theta.shape[1]
    # Matrix T
    T = np.zeros((M, q))
    for m in range(M):
        for n in range(N):
            grad_kernel = grad_kernel_tensor[m, n]
            grad_h = grad_theta_h_theta(theta, Z[n])
            T[m, :] += grad_kernel.T @ grad_h
        T[m, :] /= N
    # Matrix D(theta)
    D = np.diag([np.linalg.norm(T[:, i]) ** 2 for i in range(q)])
    # Estimation of inverse metric matrix for Wasserstein natural gradient
    D_inv = np.linalg.inv(D)
    matrix = T @ D_inv @ T.T + lamb * eps * K + eps / N * C @ C.T
    pseudo_inv = np.linalg.pinv(matrix)
    G_inv_hat = 1 / eps * (D_inv - D_inv @ T.T @ pseudo_inv @ T @ D_inv)
    return G_inv_hat

def compute_exact_wng(theta, real_theta):
    """
    Compute the exact Wasserstein natural gradient
    :param theta:
    :param real_theta:
    :return:
    """
    q = theta.shape[0] * theta.shape[1]
    d = theta.shape[1]
    exact_wng = np.zeros(q)
    exact_wng[:d] = 2 * (theta[0] - real_theta[0])
    for i in range(d):
        exact_wng[d + i] = 4 * np.sqrt(theta[1, i]) * (np.sqrt(theta[1, i]) - np.sqrt(real_theta[1, i]))
    return exact_wng


"""
Testing estimation of KWNG
"""
if __name__ == '__main__':
    import math
    # Parameters of experiment
    d = 1
    N = 1000
    M = math.floor(d * np.sqrt(N))
    lamb = 1e-10
    eps = 1e-10

    # Parameter of a random model
    np.random.seed(0)
    mu = np.random.rand(d)
    delta = np.random.rand(d)
    theta = np.array([mu, delta])
    q = theta.shape[0] * theta.shape[1]

    # Data set
    real_mu = 2 * np.zeros(d) - 1
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])

    # Samples
    print('Sampling...')
    Z = np.random.normal(size=(N, d))
    X = np.array([h_theta(theta, z) for z in Z])
    Y = X[np.random.randint(N, size=M)]
    idx = np.random.randint(d, size=M)
    print('Done!\n')

    # Gaussian kernel
    sigma_0 = 5
    dist = np.array([[np.linalg.norm(X[n] - Y[m]) for n in range(N)] for m in range(M)])
    sigma_N_M = np.mean(dist)
    gaussian_kernel_sigma = sigma_0 * sigma_N_M
    print(f'Gaussian kernel sigma: {gaussian_kernel_sigma}')

    # Kernelized Wasserstein Natural Gradient
    print('\nEstimating Kernelized Wasserstein Natural Gradient...')
    C, K, grad_kernel_tensor = compute_element_for_inverse_metric(X, Y, idx, gaussian_kernel_sigma)
    G_inv_hat = estimate_wasserstein_inverse_metric(theta, X, Y, Z, lamb, eps, C, K, grad_kernel_tensor)
    print('Done!\n')
    kwng_hat = G_inv_hat @ grad_L_theta(theta, real_theta)
    print(f'Estimated Kernelized Wasserstein natural gradient:\n{kwng_hat}')