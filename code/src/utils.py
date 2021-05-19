import math
import numpy as np


def sigmoid(x, center, z):
    return 1 / (1 + math.exp(-z * (x - center)))


class NormalGamma:
    def __init__(self, mu, v, alpha, beta):
        self.mu = mu
        self.v = v
        self.alpha = alpha
        self.beta = beta

    def update_params(self, mu, v, alpha, beta):
        self.mu = mu
        self.v = v
        self.alpha = alpha
        self.beta = beta

    def sample(self):
        tau = np.random.gamma(self.alpha, 1/self.beta)
        x = np.random.normal(self.mu, 1 / (self.v * tau))

    def sample_zhu_tan(self):
        tau = np.random.gamma(self.alpha, self.beta)
        theta = np.random.normal(self.mu, 1/self.v)

        return tau, theta
