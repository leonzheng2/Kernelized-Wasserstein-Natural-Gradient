"""
Implementation of the estimator of Kernelized Fisher-Rao Natural Gradient, with Gaussian kernel
Leon Zheng
"""

import numpy as np
from src.multinormal import h_theta, grad_theta_h_theta, grad_L_theta


class Estimator_KFRNG:
    """
    Estimator of KFRNG.
    """
    def __init__(self, N, M, d, kernel, lamb, eps):
        self.Z = np.random.normal(size=(N, d))
        self.M = M
        self.lamb = lamb
        self.eps = eps
        self.kernel = kernel

    def estimate_inv_metric(self, theta):
        """
        Estimate the inverse of Fisher-Rao information matrix.
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

        # Matrix A
        A = np.zeros((self.M, N))
        for m in range(self.M):
            for n in range(N):
                A[m, n] = - (Y[m] - X[n])[idx[m]] * self.kernel.function(Y[m], X[n])
        # Matrix K
        K = np.zeros((self.M, self.M))
        for m1 in range(self.M):
            for m2 in range(self.M):
                K[m1, m2] = self.kernel.gradient(idx[m1], idx[m2], Y[idx[m1]], Y[idx[m2]])
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

        # Estimation of inverse metric matrix for Fisher-Rao natural gradient
        D_inv = np.linalg.inv(D)
        matrix = T @ D_inv @ T.T + self.lamb * self.eps * K + self.eps / N * A @ A.T
        pseudo_inv = np.linalg.pinv(matrix)
        G_inv_hat = 1 / self.eps * (D_inv - D_inv @ T.T @ pseudo_inv @ T @ D_inv)
        return G_inv_hat

    def estimate_ng(self, theta, real_theta):
        """
        Compute the estimator of Fisher-Rao natural gradient.
        :param theta:
        :param real_theta:
        :return:
        """
        return self.estimate_inv_metric(theta) @ grad_L_theta(theta, real_theta)

def several_estimation(N_run, real_theta, N, M, d, kernel, lamb, eps, verbose=False):
    errors_list = []
    for i in range(N_run):
        # Parameter of a random model
        mu = np.random.rand(d)
        delta = np.random.rand(d)
        theta = np.array([mu, delta])

        # Estimation
        if verbose:
            print(f'\n=== Runing estimation {i}/{N_run}...===')
        estimator_kwng = Estimator_KFRNG(N, M, d, kernel, lamb, eps)
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
        errors_list.append(error)
    return np.array(errors_list)

def compute_fisher_information_matrix(theta):
    return np.diag(np.append(1 / theta[1], 1 / (2 * theta[1]**2)))

def compute_exact_ng(theta, real_theta):
    return np.linalg.inv(compute_fisher_information_matrix(theta)) @ grad_L_theta(theta, real_theta)

def compute_relative_error(kng_hat, exact_ng):
    return np.linalg.norm(kng_hat - exact_ng) / np.linalg.norm(exact_ng)


"""
Testing estimation of KFRNG
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