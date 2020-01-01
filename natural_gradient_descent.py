"""
Natural Gradient descent implementation
Leon Zheng
"""

import numpy as np
import math
from kwng import compute_exact_kwng, L_theta, estimate_wasserstein_inverse_metric, h_theta, compute_element_for_inverse_metric, grad_L_theta
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7,4))

# Parameters
d = 2
N = 1000
M = math.floor(d * np.sqrt(N))
lamb = 1e-10
eps = 1e-10
sigma_0 = 5 # Gaussian kernel
N_iter = 10 # Natural gradient
alpha = 0.45

# Data set
real_mu = np.zeros(d)
real_delta = 0.5 * np.ones(d)
real_theta = np.array([real_mu, real_delta])
q = real_theta.shape[0] * real_theta.shape[1]

# Initial theta
mu_0 = np.random.rand(d) - 0.5
delta_0 = np.random.rand(d)
theta_0 = np.array([mu_0, delta_0])

# Euclidean gradient descent
lost_euclidean = []
theta = np.array(theta_0)
print(f'Eucliedean gradient descent...\n')
for i in range(N_iter):
    print(f'Iteration {i}/{N_iter}')
    euclidean_gradient = grad_L_theta(theta, real_theta).reshape(2, d)
    theta -= alpha * euclidean_gradient
    lost_euclidean.append(L_theta(theta, real_theta))
ax.semilogy(lost_euclidean, label='Euclidean gradient')


# Natural gradient - exact computation
lost_exact_ng = []
theta = np.array(theta_0)
print(f'Natural gradient descent using exact KWNG...\n')
for i in range(N_iter):
    print(f'Iteration {i}/{N_iter}')
    natural_gradient = compute_exact_kwng(theta, real_theta).reshape(2, d)
    theta -= alpha * natural_gradient
    lost_exact_ng.append(L_theta(theta, real_theta))
ax.semilogy(lost_exact_ng, label='Exact KWNG')

# Kernelized Wasserstein Natural Gradient - estimated

## Samples
print('Sampling...')
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
print(f'Intialization:\n{theta}')
for i in range(N_iter):
    print(f'Iteration {i}/{N_iter}')
    G_inv_hat = estimate_wasserstein_inverse_metric(theta, X, Y, Z, lamb, eps, C, K, grad_kernel_tensor)
    kwng_hat = (G_inv_hat @ grad_L_theta(theta, real_theta))
    exact_kwng = compute_exact_kwng(theta, real_theta)
    print(f'Relative error: {np.linalg.norm(kwng_hat - exact_kwng) / np.linalg.norm(exact_kwng)}')
    theta -= alpha * kwng_hat.reshape(2, d)
    lost_estimated_ng.append(L_theta(theta, real_theta))
print(lost_estimated_ng)
ax.semilogy(lost_estimated_ng, label=f'Estimated KWNG (N={N})')

# Plot
ax.legend()
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title(f'd={d}')
plt.savefig(f'cv_kwng_N_{N}_d_{d}')
plt.show()
