"""
Implementation of gradient descent.
Leon Zheng
"""

import numpy as np
from src.multinormal import L_theta

class Gradient_Descent:
    """
    Gradient descent method, with fixed step size, and a given gradient.
    """
    def __init__(self, name, gradient, theta_0, N_iter, alpha, real_theta):
        self.name = name
        self.gradient = gradient
        self.theta_0 = theta_0
        self.N_iter = N_iter
        self.alpha = alpha
        self.real_theta = real_theta

    def optimize(self):
        """
        Find the minimizer.
        :return:
        """
        loss_list = []
        theta = np.array(self.theta_0)
        q = theta.shape[0] * theta.shape[1]
        d = q // 2
        theta_list = [np.array(theta).reshape(q)]
        print(f'\nGradient descent using {self.name}...')
        for i in range(self.N_iter):
            print(f'Iteration {i}/{self.N_iter}')
            grad = self.gradient(theta, self.real_theta).reshape(2, d)
            theta -= self.alpha * grad
            loss_list.append(L_theta(theta, self.real_theta))
            theta_list.append(np.array(theta).reshape(q))
        print('Done!')
        return np.array(loss_list), np.array(theta_list)

    def plot_optimization(self, pca, axes, lost_list, theta_list):
        """
        Plot the loss and the trajectory.
        :param pca:
        :param axes:
        :param lost_list:
        :param theta_list:
        :return:
        """
        axes[0].semilogy(lost_list, label = self.name)
        pca_theta = pca.transform(theta_list)
        axes[1].scatter(pca_theta[:, 0], pca_theta[:, 1], label=self.name, s=7)
