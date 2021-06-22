from numpy.random import Generator, default_rng

from src.CustomerClass import CustomerClass
from src.utils import sigmoid


class Distribution:
    def __init__(self, rng: Generator):
        # each class will have its own generator such that
        # if the same seed is used when the environment is created
        # the distribution will yield the same values and each distribution
        # will yield the same sequence of values even if other random events happen
        # in different orders
        self.rng = default_rng(seed=rng.integers(0, 2 ** 32))


class NewClicksDistribution(Distribution):
    def __init__(self, rng: Generator, new_clicks_c: float, new_clicks_z: float):
        super().__init__(rng=rng)
        self.new_clicks_c = new_clicks_c
        self.new_clicks_z = new_clicks_z

    def sample(self, customer_class: CustomerClass, bid: float):
        mean = self.mean(customer_class=customer_class, bid=bid)
        return self.rng.poisson(lam=mean, size=1)

    def sample_n(self, customer_class: CustomerClass, bid: float, n):
        mean = self.mean(customer_class=customer_class, bid=bid)
        return self.rng.poisson(lam=mean, size=n)

    def mean(self, customer_class: CustomerClass, bid: float):
        return self.n(customer_class) * self.v(bid)

    @staticmethod
    def n(customer_class: CustomerClass):
        return customer_class.newClicksR

    def v(self, bid: float):
        return sigmoid(bid, self.new_clicks_c, self.new_clicks_z)


class CostPerClickDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    @staticmethod
    def sample(customer_class: CustomerClass, bid: float):
        return bid

    @staticmethod
    def sample_n(customer_class: CustomerClass, bid: float, n: int):
        return [bid]*n

    @staticmethod
    def mean(customer_class: CustomerClass, bid: float):
        return bid


class FutureVisitsDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customer_class: CustomerClass):
        mean = self.mean(customer_class=customer_class)
        return self.rng.poisson(lam=mean, size=1)

    def sample_n(self, customer_class: CustomerClass, n):
        mean = self.mean(customer_class=customer_class)
        return self.rng.poisson(lam=mean, size=n)

    @staticmethod
    def mean(customer_class: CustomerClass):
        return customer_class.backMean


class ClickConvertedDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customer_class: CustomerClass, price: float):
        return self.rng.binomial(1, self.mean(customer_class=customer_class, price=price))

    def sample_n(self, customer_class: CustomerClass, price: float, n: int):
        return self.rng.binomial(n, self.mean(customer_class=customer_class, price=price))

    @staticmethod
    def mean(customer_class: CustomerClass, price: float):
        return sigmoid(-price, -customer_class.crCenter, customer_class.sigmoidZ)
