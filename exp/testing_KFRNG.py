"""
Testing the implementation of KFRNG.
Leon Zheng
"""

import math
import numpy as np
import src.kernel as Kernel
import matplotlib.pyplot as plt
from src.KFRNG import several_estimation

fig, ax = plt.subplots(figsize=(8, 4))

# Parameters of experiment
N = 1000
lamb = 1e-10
eps = 1e-10
scale = np.logspace(-4, -1, 10)
d_max = 1

for d in range(1, d_max+1):
    print(f'Dimension {d}...')
    M = math.floor(d * np.sqrt(N))
    # Data set
    real_mu = 2 * np.zeros(d) - 1
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])

    mean_list = []
    for sigma_0 in scale:
        kernel = Kernel.Gaussian(sigma_0, 1)
        # Estimation of the kernelized Fisher-Rao natural gradient.
        N_run = 10
        mean_list.append(np.mean(several_estimation(N_run, real_theta, N, M, d, kernel, lamb, eps)))
    ax.loglog(scale, mean_list, label=f'd={d}')

ax.grid()
ax.legend()
ax.set_xlabel('$\sigma_0$')
ax.set_ylabel('Relative error')
ax.set_title(f'N={N}' + ', M=floor(d$\sqrt{N}$)')
plt.savefig('../Test-Figures/testing_KFRNG')
plt.show()