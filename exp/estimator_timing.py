"""
Compare the timing for computing the estimator KWNG.
Leon Zheng
"""

import matplotlib.pyplot as plt
import src.kernel as Kernel
import numpy as np
import math
import time
from src.KWNG import one_estimation


fig, axes = plt.subplots(1, 2, sharey = True, figsize=(10, 4))


# Parameters of experiment
N_max = 1000
N_list = [100, 400, 700, 1000]
lamb = 1e-10
eps = 1e-10
sigma_0 = 6
d_max = 7
kernel = Kernel.Gaussian(sigma_0, 1)

#### Timing w.r.t M ####
d = 1
# Data set
real_mu = 2 * np.zeros(d) - 1
real_delta = 0.5 * np.ones(d)
real_theta = np.array([real_mu, real_delta])

M_list = [math.floor(M_) for M_ in np.linspace(10, N_max, 5)]
time_list_M = []
for M in M_list:
    t = time.time()
    print(f'Relative error: {one_estimation(real_theta, N_max, M, d, kernel, lamb, eps, verbose=False)}')
    time_list_M.append(time.time() - t)
axes[0].plot(M_list, time_list_M, label=f'd={d}, N={N_max}')
axes[0].grid()
axes[0].legend()
axes[0].set_xlabel('M')
axes[0].set_ylabel('Time (s)')
axes[0].set_title(f'N={N_max}')


#### Timing w.r.t. d ####
for N in N_list:
    M = math.floor(d * np.sqrt(N))
    time_list_d = []
    for d in range(1, d_max+1):
        # Data set
        real_mu = 2 * np.zeros(d) - 1
        real_delta = 0.5 * np.ones(d)
        real_theta = np.array([real_mu, real_delta])
        # Estimation
        t = time.time()
        print(f'Relative error: {one_estimation(real_theta, N, M, d, kernel, lamb, eps, verbose=False)}')
        time_list_d.append(time.time() - t)
    axes[1].plot(range(1, d_max+1), time_list_d, label=f'N={N}')
axes[1].grid()
axes[1].legend()
axes[1].set_xlabel('d')
axes[1].set_ylabel('Time (s)')
axes[1].set_title('M=floor($\sqrt{N}$)')

plt.subplots_adjust(hspace=0.4)
plt.savefig('../Test-Figures/estimator_timing')
plt.show()
