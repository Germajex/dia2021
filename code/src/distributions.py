from src.CustomerClass import CustomerClass
from src.utils import sigmoid
from numpy.random import Generator, default_rng


class Distribution:
    def __init__(self, rng: Generator):
        # each class will have its own generator such that
        # if the same seed is used when the environment is created
        # the distribution will yield the same values and each distribution
        # will yield the same sequence of values even if other random events happen
        # in different orders
        self.rng = default_rng(seed=rng.integers(0, 2**32))


class NewClicksDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, rng:Generator, customerClass: CustomerClass, bid: float):
        return -1

    def mean(self, customerClass: CustomerClass, bid: float):
        return customerClass.newClicksR*sigmoid(bid, customerClass.newClicksC, customerClass.newClicksZ)


class CostPerClickDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customerClass: CustomerClass, bid: float):
        return bid

    def mean(self, customerClass: CustomerClass, bid: float):
        return bid


class FutureVisitsDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customerClass: CustomerClass):
        return -1

    def mean(self, customerClass: CustomerClass):
        return customerClass.backMean


class ClickConvertedDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customerClass: CustomerClass, price: float):
        return self.rng.binomial(1, self.mean(customerClass=customerClass, price=price))

    def sample_n(self, customerClass: CustomerClass, price: float, n):
        return self.rng.binomial(n, self.mean(customerClass=customerClass, price=price))

    def mean(self, customerClass: CustomerClass, price: float):
        return sigmoid(-price, -customerClass.crCenter, customerClass.sigmoidZ)