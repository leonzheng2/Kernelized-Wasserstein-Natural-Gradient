import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(8, 4))

# Parameters
N_list = [10, 20, 30, 40, 50, 70, 100, 150, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000]
max_dim = 4

# List
for d in range(1, max_dim + 1):
    error_mean_list = np.load(f'../Test-Figures/estimator_consistency_d_{d}.npy')

    axes.plot(N_list, error_mean_list, label=f'd={d}')

axes.set_xlabel('N')
axes.set_ylabel('Relative error')
axes.set_ylim(0, 2)
axes.legend()
axes.grid()
axes.set_title('M = floor(d $\sqrt{N}$)')
plt.savefig('../Test-Figures/estimator_consistency')
plt.show()