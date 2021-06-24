import math
import numpy as np
from numpy.random import Generator


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

    def sample_zhu_tan(self):
        tau = np.random.gamma(self.alpha, self.beta)
        theta = np.random.normal(self.mu, 1 / self.v)

        return tau, theta


class Beta:
    def __init__(self, alpha, beta, rng: Generator):
        self.alpha = alpha
        self.beta = beta
        self.rng = rng

    def sample(self):
        return self.rng.beta(self.alpha, self.beta)

    def update_params(self, successes, failures):
        self.alpha += successes
        self.beta += failures

    def mean(self):
        return self.alpha / (self.alpha+self.beta)