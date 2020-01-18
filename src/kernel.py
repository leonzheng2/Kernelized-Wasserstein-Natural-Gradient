"""
Implementation of Gaussian kernels.
Leon Zheng
"""

import numpy as np

def compute_mean_dist(X, Y):
    """
    Compute the mean distance between the samples and the basis points.
    :param X:
    :param Y:
    :return:
    """
    N = X.shape[0]
    M = Y.shape[0]
    dist = np.array([[np.linalg.norm(X[n] - Y[m]) for n in range(N)] for m in range(M)])
    return np.mean(dist)

class Gaussian:
    """
    Gaussian kernel.
    """
    def __init__(self, sigma_0, s):
        self.sigma_0 = sigma_0
        self.s = s
        self.sigma = sigma_0

    def set_param(self, X, Y):
        """
        Set value to sigma, the parameter of the kernel.
        :param X:
        :param Y:
        :return:
        """
        self.sigma = self.sigma_0 * compute_mean_dist(X, Y)

    def function(self, a, b):
        """
        Gaussian kernel
        :param a: vector (d, )
        :param b: vector (d, )
        :param sigma: float
        :return: float
        """
        return self.s * np.exp(- np.linalg.norm(a - b) ** 2 / (2 * self.sigma ** 2))

    def gradient(self, i, j, a, b):
        """
        Compute \partial_k \partial_{l+d} k(a, b), for k, l \in [1,d]
        :param k: int in [1,d]
        :param l: int in [1,d]
        :param a: vector (d, )
        :param b: vector (d, )
        :param sigma: float
        :return: float
        """
        return 1 / self.sigma ** 2 * self.s * np.exp(- np.linalg.norm(a - b) ** 2 / (2 * self.sigma ** 2)) * (
                    -1 / self.sigma ** 2 * (a[i] - b[i]) * (a[j] - b[j]) + (i == j))

class Laplacian:
    def __init__(self, sigma_0, s):
        self.sigma_0 = sigma_0
        self.s = s
        self.sigma = sigma_0

    def set_param(self, X, Y):
        self.sigma = self.sigma_0 * compute_mean(X, Y)

    def function(self, a, b):
        return self.s * np.exp(- np.linalg.norm(a - b) / self.sigma)

    def gradient(self, i, j, a, b):
        norm = np.linalg.norm(a - b)
        # print(self.sigma)
        print(norm < 1e10)
        return - 1 / (self.sigma * norm**2) * self.function(a, b) * ( - norm * (i==j) + (a[i] - b[i]) * (a[j] - b[j]) * (1 / self.sigma + 1 / norm))


class Sigmoid:
    def __init__(self, s0):
        self.s0 = s0
        self.threshold = 1e10

    def set_param(self, X, Y):
        N = X.shape[0]
        M = Y.shape[0]
        dot_product = np.array([[X[n] @ Y[m] for n in range(N)] for m in range(M)])
        self.alpha = self.s0 * 1 / np.std(dot_product)
        self.c = - self.s0 * np.mean(dot_product) / np.std(dot_product)

    def function(self, x, y):
        k = np.tanh(self.alpha * x @ y + self.c)
        if k < self.threshold:
            return self.threshold
        return k

    def gradient(self, i, j, x, y):
        return (1 - self.function(x, y)**2) * (self.alpha * (i == j) - 2 * self.function(x, y) * (self.alpha * x[j] + self.c) * (self.alpha * y[i] + self.c))

if __name__ == '__main__':
    kernel = Sigmoid(1, 0)
    x = -1 * np.ones(5)
    print(kernel.gradient(0, 0, x, x))
