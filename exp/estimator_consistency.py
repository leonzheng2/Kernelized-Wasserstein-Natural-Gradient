"""
Experiments for convergence of the estimate of KWNG. Multivariate normal model. Consistency.
Leon Zheng
"""

from src.KWNG import several_estimation_without_outliers
import math
import numpy as np
import matplotlib.pyplot as plt
import src.kernel as Kernel

fig, axes = plt.subplots(figsize=(8, 4))

# Parameters
N_list = [10, 20, 30, 40, 50, 70, 100, 150, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000]
max_dim = 4
sigma_0 = 5
s = 1
lamb = 1e-10
eps = 1e-10
N_run = 10
max_run = 3 * N_run
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
    for N in N_list:
        M = math.floor(d * np.sqrt(N))
        print(f'\n############### N={N}, M={M}, d={d} ###############')
        errors = several_estimation_without_outliers(N_run, max_run, threshold, real_theta, N, M, d, kernel, lamb, eps)
        error_std_list.append(np.std(errors))
        error_mean_list.append(np.mean(errors))
    np.save(f'../Test-Figures/estimator_consistency_d_{d}', error_mean_list)
    axes.plot(N_list, error_mean_list, label=f'd={d}')

axes.set_xlabel('N')
axes.set_ylabel('Relative error')
axes.set_ylim(0, 1)
axes.legend()
axes.grid()
axes.set_title('M = floor(d $\sqrt{N}$)')
plt.savefig('../Test-Figures/estimator_consistency')
plt.show()