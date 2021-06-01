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
        raise NotImplementedError

    @staticmethod
    def mean(customer_class: CustomerClass, bid: float):
        return round(customer_class.newClicksR * sigmoid(bid, customer_class.newClicksC, customer_class.newClicksZ))


class CostPerClickDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, bid: float):
        return bid

    @staticmethod
    def mean(bid: float):
        return bid


class FutureVisitsDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customerClass: CustomerClass):
        raise NotImplementedError

    @staticmethod
    def mean(customer_class: CustomerClass):
        return customer_class.backMean


class ClickConvertedDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customer_class: CustomerClass, price: float):
        return self.rng.binomial(1, self.mean(customer_class=customer_class, price=price))

    def sample_n(self, customer_class: CustomerClass, price: float, n):
        return self.rng.binomial(n, self.mean(customer_class=customer_class, price=price))

    @staticmethod
    def mean(customer_class: CustomerClass, price: float):
        return sigmoid(-price, -customer_class.crCenter, customer_class.sigmoidZ)