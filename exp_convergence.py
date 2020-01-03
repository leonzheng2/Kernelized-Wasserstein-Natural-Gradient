"""
Experiments for convergence of the estimate of KWNG. Multivariate normal model.
Leon Zheng
"""

import math
import numpy as np
from kwng import estimate_wasserstein_inverse_metric, compute_element_for_inverse_metric
from multi_normal_model import grad_L_theta, h_theta

def experience_relative_errors(d, N_list, sigma_0, num_run, real_theta, lamb, eps):
    errors_list = []
    for run in range(num_run):
        print(f'======== Run {run + 1}/{num_run} ========')

        # Parameter of a random model
        mu = 2 * np.random.rand(d) - 1
        delta = np.random.rand(d)
        theta = np.array([mu, delta])
        q = theta.shape[0] * theta.shape[1]

        # Exact Natural Wassertsein gradient
        print('Exact Wasserstein natural gradient:')
        exact_wng = np.zeros(q)
        exact_wng[:d] = 2 * (theta[0] - real_theta[0])
        for i in range(d):
            exact_wng[d + i] = 4 * np.sqrt(delta[i]) * (np.sqrt(delta[i]) - np.sqrt(real_theta[1, i]))
        print(f'{exact_wng}')

        # Sampling data
        count = 0
        N = N_list[count]
        M = math.floor(d * np.sqrt(N))
        Z = np.random.normal(size=(N, d))
        X = np.array([h_theta(theta, z) for z in Z])
        Y = X[np.random.randint(N, size=M)]
        idx = np.random.randint(d, size=M)

        kwng_hat_list = []
        while count < len(N_list):
            # Gaussian kernel
            dist = np.array([[np.linalg.norm(X[n] - Y[m]) for n in range(N)] for m in range(M)])
            sigma_N_M = np.mean(dist)
            gaussian_kernel_sigma = sigma_0 * sigma_N_M
            # print(f'Gaussian kernel sigma: {gaussian_kernel_sigma}\n')

            # Kernelized Wasserstein Natural Gradient
            # print(f'Estimating Kernelized Wasserstein Natural Gradient, with {N} samples...')
            C, K, grad_kernel_tensor = compute_element_for_inverse_metric(X, Y, idx, gaussian_kernel_sigma)
            G_inv_hat = estimate_wasserstein_inverse_metric(theta, X, Y, Z, lamb, eps, C, K, grad_kernel_tensor)
            # print('Done!\n')
            kwng_hat = G_inv_hat @ grad_L_theta(theta, real_theta)
            # print(f'Estimated Kernelized Wasserstein natural gradient:\n{kwng_hat}')
            kwng_hat_list.append(kwng_hat)

            # Sampling
            count += 1
            if count == len(N_list):
                break
            new_Z = np.random.normal(size=(N_list[count] - N_list[count - 1], d))
            Z = np.append(Z, new_Z, axis=0)
            X = np.append(X, np.array([h_theta(theta, z) for z in new_Z]), axis=0)
            N = N_list[count]
            new_M = math.floor(d * np.sqrt(N))
            if new_M > M:
                new_Y = X[np.random.randint(N, size=new_M - M)]
                Y = np.append(Y, new_Y, axis=0)
                new_idx = np.random.randint(d, size=new_M - M)
                idx = np.append(idx, new_idx)
                M = new_M

        # Convergence: relative error
        print(f'Estimated Kernelized Wasserstein Natural Gradient:\n{kwng_hat_list[-1]}\n')
        errors = [np.linalg.norm(kwng_hat - exact_wng) / np.linalg.norm(exact_wng) for kwng_hat in kwng_hat_list]
        errors_list.append(errors)
    return errors_list

if __name__ == '__main__':
    # Parameters of experiment
    d = 2
    lamb = 1e-10
    eps = 1e-10
    num_run = 10
    N_list = [1000]
    sigma_0 = 5

    # Data set
    real_mu = np.zeros(d)
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])

    # Experience
    e = experience_relative_errors(d, N_list, sigma_0, num_run, real_theta, lamb, eps)
    print(e[-1])