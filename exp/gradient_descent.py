"""
Experiments comparing different gradient descent methods for Gaussian model.
Leon Zheng
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src import KFRNG, KWNG
from src.gradient_descent import Gradient_Descent
import numpy as np
from src.multinormal import grad_L_theta
import math
import src.kernel as Kernel

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pca = PCA(n_components=2)

######### PARAMETERS ########
# Problem dimensions
d = 2
# Gradient descent
N_iter = 800
alpha = 0.1
# Estimator
N = 128
# M = math.floor(d * np.sqrt(N))
M = 100
lamb = 1e-10
eps = 1e-10
sigma_0 = 5  # Gaussian kernel
s = 1
kernel = Kernel.Gaussian(sigma_0, s)
##############################

# Data set
real_mu = np.zeros(d)
real_delta = np.ones(d)
real_theta = np.array([real_mu, real_delta])
q = real_theta.shape[0] * real_theta.shape[1]

# Initial theta
mu_0 = np.random.rand(d)
delta_0 = np.random.rand(d)
theta_0 = np.array([mu_0, delta_0])
# theta_0 = np.array([[0.56091229, 0.51912719], [0.91251602, 0.54109815]])
print(f'Initialization:\n{theta_0}')

# Exact WNG
wng_descent = Gradient_Descent('WNG', KWNG.compute_exact_ng, theta_0, N_iter, alpha, real_theta)
wng_lost_list, wng_theta_list = wng_descent.optimize()
pca.fit(wng_theta_list)  # PCA for trajectories plot
wng_descent.plot_optimization(pca, axes, wng_lost_list, wng_theta_list)

# Estimated KWNG
estimator_KWNG = KWNG.Estimator_KWNG(N, M, d, kernel, lamb, eps)
kwng_descent = Gradient_Descent(f'KWNG (N={N}, M={M})', estimator_KWNG.estimate_ng, theta_0, N_iter, alpha, real_theta)
kwng_lost_list, kwng_theta_list = kwng_descent.optimize()
kwng_descent.plot_optimization(pca, axes, kwng_lost_list, kwng_theta_list)

# Euclidean gradient
euclidean_descent = Gradient_Descent('GD', grad_L_theta, theta_0, N_iter, alpha, real_theta)
gd_lost_list, gd_theta_list = euclidean_descent.optimize()
euclidean_descent.plot_optimization(pca, axes, gd_lost_list, gd_theta_list)

# Exact FRNG
frng_descent = Gradient_Descent('FRNG', KFRNG.compute_exact_ng, theta_0, N_iter, alpha, real_theta)
frng_lost_list, frng_theta_list = frng_descent.optimize()
frng_descent.plot_optimization(pca, axes, frng_lost_list, frng_theta_list)

# Legend, plot, title, save
plt.subplots_adjust(wspace=0.4)
axes[0].legend()
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Loss')
axes[0].set_title(f'd={d}')
axes[0].grid()
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title(f'd={d}, alpha={alpha}')
axes[1].grid()
plt.savefig(f'../Test-Figures/cv_kwng_N_{N}_d_{d}')
plt.show()