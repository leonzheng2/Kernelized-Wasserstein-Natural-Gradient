"""
Experiments for convergence of the estimate of KWNG. Multivariate normal model. Number of basis points.
Leon Zheng
"""

from src.KWNG import several_estimation_without_outliers
import math
import numpy as np
import matplotlib.pyplot as plt
import src.kernel as Kernel

fig, axes = plt.subplots(figsize=(8, 4))

# Parameters
N = 1000
max_dim = 4
M_list = [50, 100, 150, 200, 300, 500]
sigma_0 = 5
s = 1
lamb = 1e-10
eps = 1e-10
N_run = 10
max_run = 30
threshold = 10
kernel = Kernel.Gaussian(sigma_0, s)


for d in range(1, max_dim + 1):
    # Data set
    real_mu = 2 * np.zeros(d) - 1
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])

    # Estimation error for several values of N.
    error_std_list = []
    error_mean_list = []
    for M in M_list:
        print(f'\n############### N={N}, M={M}, d={d} ###############')
        errors = several_estimation_without_outliers(N_run, max_run, threshold, real_theta, N, M, d, kernel, lamb, eps)
        error_std_list.append(np.std(errors))
        error_mean_list.append(np.mean(errors))
    np.save(f'../Test-Figures/estimator_basis_d_{d}', error_mean_list)
    axes.plot(M_list, error_mean_list, label=f'd={d}')

axes.set_xlabel('M')
axes.set_ylabel('Relative error')
axes.set_ylim(0, 2)
axes.legend()
axes.grid()
axes.set_title(f'N={N}')
plt.savefig('../Test-Figures/estimator_basis')
plt.show()