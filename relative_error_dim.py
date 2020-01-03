"""
Experience comparing several dimensions.
Leon Zheng
"""

from exp_convergence import experience_relative_errors
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))

for d in range(1, 5):
    # Parameters of experiment
    N_max = 4000
    lamb = 1e-10
    eps = 1e-10
    sigma_0 = 5  # Gaussian kernel sigma
    num_run = 5
    N_list = [10, 100, 200, 400, 1000, 2000, 4000]

    # Data set
    real_mu = np.zeros(d)
    real_delta = 0.5 * np.ones(d)
    real_theta = np.array([real_mu, real_delta])

    # Convergence: compute relative errors
    errors_list = experience_relative_errors(d, N_list, sigma_0, num_run, real_theta, lamb, eps)
    np.save(f'errors_list_d_{d}', errors_list)

    # Average over runs for relative errors
    inliers = []
    for error in errors_list:
        # print(error[-1])
        if error[-1] < 1e2:
            inliers.append(error)
    error_list_inliers = np.array(inliers)
    errors_mean = np.mean(error_list_inliers, axis=0)
    print(errors_mean)

    ax.plot(N_list, errors_mean, label=f'd={d}')

ax.set_xlabel('N')
ax.set_ylabel('Relative error')
plt.savefig('relative_error')
plt.show()