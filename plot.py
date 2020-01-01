import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
N_list = [10, 100, 200, 400, 1000, 2000, 4000]

for d in range(1, 4):
    # Convergence: compute relative errors
    errors_list = np.load(f'errors_list_d_{d}.npy')

    # Average over runs for relative errors
    inliers = []
    for error in errors_list:
        # print(error[-1])
        if error[-1] < 1e2:
            inliers.append(error)
    error_list_inliers = np.array(inliers)
    errors_mean = np.mean(error_list_inliers, axis=0)
    print(errors_mean)

    if d < 3:
        ax.plot(N_list, errors_mean, label=f'd={d}')
    else:
        ax.plot(N_list[1:], errors_mean[1:], label=f'd={d}')

ax.set_ylim(0, 0.3)
ax.set_xlabel('N')
ax.set_ylabel('Relative error')
ax.legend()
plt.savefig('relative_error')
plt.show()