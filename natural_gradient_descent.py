"""
Natural Gradient descent implementation
Leon Zheng
"""

import numpy as np
import math
from kwng import compute_exact_kwng, L_theta, estimate_wasserstein_inverse_metric, h_theta, compute_element_for_inverse_metric, grad_L_theta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
pca = PCA(n_components=2)

# Parameters
d = 2
N = 128
# M = math.floor(d * np.sqrt(N))
M = 100
lamb = 1e-10
eps = 1e-10
sigma_0 = 5 # Gaussian kernel
N_iter = 1000 # Natural gradient
alpha = 0.1

# Data set
real_mu = np.zeros(d)
real_delta = np.ones(d)
real_theta = np.array([real_mu, real_delta])
q = real_theta.shape[0] * real_theta.shape[1]

# Initial theta
mu_0 = np.random.rand(d)
delta_0 = np.random.rand(d)
theta_0 = np.array([mu_0, delta_0])
# theta_0 = np.array([[0.71980233, 0.09748811, 0.97565858], [0.54323799, 0.17849202, 0.6167477 ]])
print(f'Initialization:\n{theta_0}')

# Natural gradient - exact computation
lost_exact_ng = []
theta = np.array(theta_0)
theta_exact_ng = [np.array(theta).reshape(q)]
print(f'\nNatural gradient descent using exact KWNG...')
for i in range(N_iter):
    print(f'Iteration {i}/{N_iter}')
    natural_gradient = compute_exact_kwng(theta, real_theta).reshape(2, d)
    theta -= alpha * natural_gradient
    lost_exact_ng.append(L_theta(theta, real_theta))
    theta_exact_ng.append(np.array(theta).reshape(q))

# Euclidean gradient descent
lost_euclidean = []
theta = np.array(theta_0)
theta_euclidean = [np.array(theta).reshape(q)]
print(f'Eucliedean gradient descent...')
for i in range(N_iter):
    print(f'Iteration {i}/{N_iter}')
    euclidean_gradient = grad_L_theta(theta, real_theta).reshape(2, d)
    theta -= alpha * euclidean_gradient
    lost_euclidean.append(L_theta(theta, real_theta))
    theta_euclidean.append(np.array(theta).reshape(q))

# Kernelized Wasserstein Natural Gradient - estimated

## Samples
print('\nSampling...')
Z = np.random.normal(size=(N, d))
X = np.array([h_theta(theta, z) for z in Z])
Y = X[np.random.randint(N, size=M)]
idx = np.random.randint(d, size=M)
print('Done!\n')

## Gaussian kernel
print('Gaussian kernel...')
dist = np.array([[np.linalg.norm(X[n] - Y[m]) for n in range(N)] for m in range(M)])
sigma_N_M = np.mean(dist)
gaussian_kernel_sigma = sigma_0 * sigma_N_M
print(f'Gaussian kernel sigma: {gaussian_kernel_sigma}\n')

## Elements for inverse metric
print(f'Computing elements for inversing metric...')
C, K, grad_kernel_tensor = compute_element_for_inverse_metric(X, Y, idx, gaussian_kernel_sigma)
print(f'Natural gradient descent using estimated KWNG...')

## Natural gradient descent
lost_estimated_ng = []
theta = np.array(theta_0)
theta_estimated_ng = [np.array(theta).reshape(q)]
print(f'Intialization:\n{theta}')
for i in range(N_iter):
    print(f'Iteration {i}/{N_iter}')
    G_inv_hat = estimate_wasserstein_inverse_metric(theta, X, Y, Z, lamb, eps, C, K, grad_kernel_tensor)
    kwng_hat = (G_inv_hat @ grad_L_theta(theta, real_theta))
    exact_kwng = compute_exact_kwng(theta, real_theta)
    print(f'Relative error: {np.linalg.norm(kwng_hat - exact_kwng) / np.linalg.norm(exact_kwng)}')
    theta -= alpha * kwng_hat.reshape(2, d)
    lost_estimated_ng.append(L_theta(theta, real_theta))
    theta_estimated_ng.append(np.array(theta).reshape(q))


# Plot

# PCA for trajectories plot
pca.fit(theta_exact_ng)

# Exact WNG
axes[0].semilogy(lost_exact_ng, label='WNG')
theta_exact_ng = np.array(theta_exact_ng)
print(theta_exact_ng)
pca_exact_ng = pca.transform(theta_exact_ng)
print(pca_exact_ng)
axes[1].scatter(pca_exact_ng[:, 0], pca_exact_ng[:, 1], label='WNG', s=7)

# Estimated KWNG
print(lost_estimated_ng)
axes[0].semilogy(lost_estimated_ng, label=f'KWNG (N={N})')
theta_estimated_ng = np.array(theta_estimated_ng)
pca_estimated_ng = pca.transform(theta_estimated_ng)
axes[1].scatter(pca_estimated_ng[:, 0], pca_estimated_ng[:, 1], label=f'KWNG (N={N})', s=7)

# Gradient descent
axes[0].semilogy(lost_euclidean, label='GD')
theta_euclidean = np.array(theta_euclidean)
pca_euclidean = pca.transform(theta_euclidean)
print(pca_euclidean)
axes[1].scatter(pca_euclidean[:, 0], theta_euclidean[:, 1], label='GD', s=7)

# Legend
axes[0].legend()
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Loss')
axes[0].set_title(f'd={d}')

# axes[1].legend()
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title(f'$d={d}$, alpha={alpha}')
# print(theta_euclidean)
# print(theta_exact_ng)
# print(theta_estimated_ng)

plt.savefig(f'cv_kwng_N_{N}_d_{d}')
plt.show()
