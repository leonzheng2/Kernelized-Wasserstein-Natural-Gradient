"""
Parametric model: multivariate normal model.
Loss function: Wasserstein 2 distance.
Leon Zheng
"""

import numpy as np
import scipy.linalg

"""
Representation
 * theta = [mu, delta]. Shape: (2, d).
 * mu is the mean of the Gaussian. Shape: (d, )
 * delta is the diagonal of the covariance matrix. Shape: (d, )
"""

def h_theta(theta, z):
    """
    Output x from latent variable z according the model of parameter theta, multivariate normal model.
    :param theta: matrix (2, d)
    :param z: vector (d, )
    :return: vector (d, )
    """
    mu = theta[0]
    delta = np.diag(theta[1])
    root = scipy.linalg.sqrtm(delta)
    return root @ z + mu

def grad_theta_h_theta(theta, z):
    """
    Output the gradient of \theta \mapsto h_{\theta}(z), which is a jacobian matrix, shape (d, q), q = 2d.
    :param theta: matrix (2, d)
    :param z: vector (d, )
    :return: matrix (d, 2d)
    """
    d = theta.shape[1]
    delta = theta[1]
    grad = np.zeros((d, 2*d))
    for i in range(d):
        grad[i, i] = 1
        grad[i, d + i] = z[i] / (2 * np.sqrt(delta[i]))
    return grad


"""
Lost function: Wasserstein 2 distance squared.
"""

def L_theta(theta, real_theta):
    """
    Loss function. Wasserstein 2 distance squared.
    :param theta:
    :param real_theta:
    :return:
    """
    return np.linalg.norm(theta[0] - real_theta[0])**2 + np.linalg.norm(np.sqrt(theta[1]) - np.sqrt(real_theta[1]))**2

def grad_L_theta(theta, real_theta):
    """
    Gradient of loss function.
    :param theta:
    :param real_theta:
    :return:
    """
    q = theta.shape[0] * theta.shape[1]
    d = theta.shape[1]
    grad = np.zeros(q)
    grad[:d] = 2 * (theta[0] - real_theta[0])
    for i in range(d):
        grad[d + i] = (np.sqrt(theta[1, i]) - np.sqrt(real_theta[1, i])) / np.sqrt(theta[1, i])
    return grad

