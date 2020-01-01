"""
Find the best Gaussian kernel in practice for convergence of estimated KWNG
Leon Zheng
"""

import numpy as np
from exp_convergence import experience_relative_errors
import matplotlib.pyplot as plt


# Parameters of experiment
d = 2
lamb = 1e-10
eps = 1e-10
num_run = 10
N_list = [1000]

# Data set
real_mu = np.zeros(d)
real_delta = 0.5 * np.ones(d)
real_theta = np.array([real_mu, real_delta])

sigma_0_list = np.linspace(4.3, 6, 10)
errors = []
for sigma_0 in sigma_0_list:
    print(f'%%%%%%%%% HYPERPARAMETER sigma_0 = {sigma_0} %%%%%%%%%')
    e = experience_relative_errors(d, N_list, sigma_0, num_run, real_theta, lamb, eps)
    errors.append(e[-1])
plt.semilogy(sigma_0_list, errors)
plt.show()
