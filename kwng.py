"""
Implementation of Kernelized Wasserstein NAtural Gradient
Leon Zheng
"""

import numpy as np
import scipy.linalg
import time

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

"""
Representation
 * theta = [mu, delta]. Shape: (2, d).
 * mu is the mean of the Gaussian. Shape: (d, )
 * delta is the diagonal of the covariance matrix. Shape: (d, )
"""

def h_theta(theta, z):
    """
    Output x from latent variable z according the model of parameter theta, multivariate normal model.
    :param theta: matrix (2, d)
    :param z: vector (d, )
    :return: vector (d, )
    """
    mu = theta[0]
    delta = np.diag(theta[1])
    root = scipy.linalg.sqrtm(delta)
    return root @ z + mu

def grad_theta_h_theta(theta, z):
    """
    Output the gradient of \theta \mapsto h_{\theta}(z), which is a jacobian matrix, shape (d, q), q = 2d.
    :param theta: matrix (2, d)
    :param z: vector (d, )
    :return: matrix (d, 2d)
    """
    d = theta.shape[1]
    delta = theta[1]
    grad = np.zeros((d, 2*d))
    for i in range(d):
        grad[i, i] = 1
        grad[i, d + i] = z[i] / (2 * np.sqrt(delta[i]))
    return grad


"""
Lost function: Wasserstein 2 distance squared.
"""

def L_theta(theta, real_theta):
    return np.linalg.norm(theta[0] - real_theta[0])**2 + np.linalg.norm(np.sqrt(theta[1]) - np.sqrt(real_theta[1]))**2

def grad_L_theta(theta, real_theta):
    q = theta.shape[0] * theta.shape[1]
    d = theta.shape[1]
    grad = np.zeros(q)
    grad[:d] = 2 * (theta[0] - real_theta[0])
    for i in range(d):
        grad[d + i] = (np.sqrt(theta[1, i]) - np.sqrt(real_theta[1, i])) / np.sqrt(theta[1, i])
    return grad


"""
Kernelized Wasserstein Natural Gradient
"""

def compute_element_for_inverse_metric(X, Y, idx, gaussian_kernel_sigma):
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


def estimate_kwng(G_inv_hat, theta, real_theta):
    # Estimation of the Kernelized Wasserstein Natural Gradient (KWNG)
    kwng_hat = G_inv_hat @ hat_grad_L_theta(theta, real_theta)
    return kwng_hat

def compute_exact_kwng(theta, real_theta):
    q = theta.shape[0] * theta.shape[1]
    d = theta.shape[1]
    exact_wng = np.zeros(q)
    exact_wng[:d] = 2 * (theta[0] - real_theta[0])
    for i in range(d):
        exact_wng[d + i] = 4 * np.sqrt(theta[1, i]) * (np.sqrt(theta[1, i]) - np.sqrt(real_theta[1, i]))
    return exact_wng

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
    kwng_hat = G_inv_hat @ hat_grad_L_theta(theta, real_theta)
    print(f'Estimated Kernelized Wasserstein natural gradient:\n{kwng_hat}')