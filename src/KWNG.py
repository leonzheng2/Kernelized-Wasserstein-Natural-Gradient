"""
Implementation of the estimator of Kernelized Wasserstein Natural Gradient, with Gaussian kernel
Leon Zheng
"""

import numpy as np
from src.multinormal import h_theta, grad_theta_h_theta, grad_L_theta


class Estimator_KWNG:
    """
    Estimator of KWNG.
    """
    def __init__(self, N, M, d, kernel, lamb, eps):
        self.Z = np.random.normal(size=(N, d))
        self.M = M
        self.lamb = lamb
        self.eps = eps
        self.kernel = kernel

    def estimate_inv_metric(self, theta):
        """
        Estimate the inverse of Wasserstein information matrix.
        :param theta:
        :return:
        """
        # Samples
        q = theta.shape[0] * theta.shape[1]
        X = np.array([h_theta(theta, z) for z in self.Z])
        N = self.Z.shape[0]
        d = X.shape[1]
        Y = X[np.random.randint(N, size = self.M)]
        idx = np.random.randint(d, size = self.M)

        # Kernel
        self.kernel.set_param(X, Y)

        # Matrix C
        C = np.zeros((self.M, N * d))
        for m in range(self.M):
            for n in range(N):
                for i in range(d):
                    C[m, n * d + i] = self.kernel.gradient(idx[m], i, Y[m], X[n])
        # print(C)
        # Matrix K
        K = np.zeros((self.M, self.M))
        for m1 in range(self.M):
            for m2 in range(self.M):
                K[m1, m2] = self.kernel.gradient(idx[m1], idx[m2], Y[idx[m1]], Y[idx[m2]])
        # print(K)
        # Matrix T
        T = np.zeros((self.M, q))
        for m in range(self.M):
            for n in range(N):
                grad_kernel = np.array([self.kernel.gradient(idx[m], i, Y[m], X[n]) for i in range(d)])
                grad_h = grad_theta_h_theta(theta, self.Z[n])
                T[m, :] += grad_kernel.T @ grad_h
            T[m, :] /= N
        # Matrix D(theta)
        D = np.diag([np.linalg.norm(T[:, i]) ** 2 for i in range(q)])

        # Estimation of inverse metric matrix for Wasserstein natural gradient
        D_inv = np.linalg.inv(D)
        matrix = T @ D_inv @ T.T + self.lamb * self.eps * K + self.eps / N * C @ C.T
        # print(matrix)
        pseudo_inv = np.linalg.pinv(matrix)
        G_inv_hat = 1 / self.eps * (D_inv - D_inv @ T.T @ pseudo_inv @ T @ D_inv)
        return G_inv_hat

    def estimate_ng(self, theta, real_theta):
        """
        Compute the estimator of Wasserstein natural gradient.
        :param theta:
        :param real_theta:
        :return:
        """
        return self.estimate_inv_metric(theta) @ grad_L_theta(theta, real_theta)

def compute_exact_ng(theta, real_theta):
    """
    Compute the exact Wasserstein natural gradient
    :param theta:
    :param real_theta:
    :return:
    """
    q = theta.shape[0] * theta.shape[1]
    d = theta.shape[1]
    exact_ng = np.zeros(q)
    exact_ng[:d] = 2 * (theta[0] - real_theta[0])
    for i in range(d):
        exact_ng[d + i] = 4 * np.sqrt(theta[1, i]) * (np.sqrt(theta[1, i]) - np.sqrt(real_theta[1, i]))
    return exact_ng

def compute_relative_error(kng_hat, exact_ng):
    """
    Computing the relative error between the estimated gradient and the exact.
    :param kng_hat:
    :param exact_ng:
    :return:
    """
    return np.linalg.norm(kng_hat - exact_ng) / np.linalg.norm(exact_ng)

def one_estimation(real_theta, N, M, d, kernel, lamb, eps, verbose=False):
    """
    Performing one estimation
    :param real_theta:
    :param N:
    :param M:
    :param d:
    :param kernel:
    :param lamb:
    :param eps:
    :param verbose:
    :return:
    """
    # Parameter of a random model
    mu = np.random.rand(d)
    delta = np.random.rand(d)
    theta = np.array([mu, delta])
    # Estimation
    estimator_kwng = Estimator_KWNG(N, M, d, kernel, lamb, eps)
    if verbose:
        print('Estimating inverse of Wasserstein information matrix...')
    G_inv_hat = estimator_kwng.estimate_inv_metric(theta)
    if verbose:
        print('Done!')
    kng_hat = G_inv_hat @ grad_L_theta(theta, real_theta)
    exact_ng = compute_exact_ng(theta, real_theta)
    if verbose:
        print(f'Estimated Kernelized Wasserstein natural gradient:\n{kng_hat}')
        print(f'Exact Wasserstein natural gradient:\n{exact_ng}')
    error = compute_relative_error(kng_hat, exact_ng)
    if verbose:
        print(f'Relative error: {error}')
    return error

def several_estimation(N_run, real_theta, N, M, d, kernel, lamb, eps, verbose=False):
    errors_list = []
    for i in range(N_run):
        if verbose:
            print(f'\n=== Runing estimation {i}/{N_run}...===')
        error = one_estimation(real_theta, N, M, d, kernel, lamb, eps, verbose=verbose)
        errors_list.append(error)
    return np.array(errors_list)

def several_estimation_without_outliers(N_run, i_max, threshold, real_theta, N, M, d, kernel, lamb, eps):
    errors_list = []
    i = 0
    j = 0
    while j < N_run and i < i_max:
        print(f'\n=== Runing {i}-th estimation: {j}/{N_run} inliers...===')
        error = one_estimation(real_theta, N, M, d, kernel, lamb, eps)
        # Outlier
        if error < threshold:
            errors_list.append(error)
            j += 1
        i += 1
    return np.array(errors_list)

def eliminate_outliers(list, thershold):
    new_list = []
    for l in list:
        if l < thershold:
            new_list.append(l)
    return np.array(new_list)


"""
Testing estimation of KWNG
"""
if __name__ == '__main__':
    import math
    import kernel as Kernel
    # Parameters of experiment
    d = 1
    N = 1000
    M = math.floor(d * np.sqrt(N))
    kernel = Kernel.Sigmoid(1e-17)
    lamb = 1e-10
    eps = 1e-10

    # Data set
    real_mu = 2 * np.zeros(d) - 1
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])

    # Estimation of the kernelized Wasserstein natural gradient.
    N_run = 10
    several_estimation(N_run, real_theta, N, M, d, kernel, lamb, eps, verbose=True)