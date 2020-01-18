"""
Experiments for convergence of the estimate of KWNG. Multivariate normal model. Several dimensions.
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
sigma_0 = 5
s = 1
lamb = 1e-10
eps = 1e-10
N_run = 10
max_run = 3 * N_run
threshold = 2
kernel = Kernel.Gaussian(sigma_0, s)

errors = []
for d in range(1, max_dim + 1):
    # Data set
    real_mu = 2 * np.zeros(d) - 1
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])
    M = math.floor(d * np.sqrt(N))

    # Estimation error, for several values run.
    print(f'\n############### N={N}, M={M}, d={d} ###############')
    err = several_estimation_without_outliers(N_run, max_run, threshold, real_theta, N, M, d, kernel, lamb, eps)
    errors.append(err)
    np.save(f'../Test-Figures/estimator_dimensions_N_{N}', errors)

axes.boxplot(errors)
axes.set_xlabel('d')
axes.set_ylabel('Relative error')
axes.set_ylim(0, threshold)
axes.grid()
axes.set_title(f'N={N}' + ', M=floor(d$\sqrt{N}$)')
plt.savefig(f'../Test-Figures/estimator_dimensions_N_{N}')
plt.show()