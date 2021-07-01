import numpy as np
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


class TotalAuctionsDistribution(Distribution):
    def __init__(self, rng: Generator, lambda_a):
        super().__init__(rng=rng)
        self.lambda_a = lambda_a

    def sample(self):
        return int(self.rng.poisson(lam=self.lambda_a, size=1)[0])


class AuctionsPerCombinationDistribution(Distribution):
    def __init__(self, rng: Generator, likelihoods_per_comb):
        super().__init__(rng=rng)
        self.combs = []
        self.likelihoods = []

        for comb, likelihood in likelihoods_per_comb.items():
            self.combs.append(comb)
            self.likelihoods.append(likelihood)

    def sample(self, tot_auctions):
        auctions = self.rng.multinomial(tot_auctions, self.likelihoods)
        res = {c: a for c, a in zip(self.combs, auctions)}
        return res


class NewClicksDistribution(Distribution):
    def __init__(self, rng: Generator, new_clicks_c: float, new_clicks_z: float, tot_auctions: int,
                 likelihoods_per_comb):
        super().__init__(rng=rng)
        self.new_clicks_c = new_clicks_c
        self.new_clicks_z = new_clicks_z
        self.tot_auction = tot_auctions
        self.likelihoods_per_comb = likelihoods_per_comb

    def sample(self, auctions_per_comb, bid: float):
        winning_p = self.v(bid)

        res = {comb: self.rng.binomial(auctions, winning_p)
               for comb, auctions in auctions_per_comb.items()}

        return res

    def mean(self, customer_class: CustomerClass, bid: float):
        return np.sum([self.likelihoods_per_comb[comb] for comb in customer_class.features]) * self.tot_auction * \
               self.v(bid)

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
        return int(CostPerClickDistribution.sample_n(customer_class, bid, 1)[0])

    @staticmethod
    def sample_n(customer_class: CustomerClass, bid: float, n: int):
        return [bid] * n

    @staticmethod
    def mean(customer_class: CustomerClass, bid: float):
        return bid


class FutureVisitsDistribution(Distribution):
    def __init__(self, rng: Generator):
        super().__init__(rng=rng)

    def sample(self, customer_class: CustomerClass):
        mean = self.mean(customer_class=customer_class)
        return int(self.rng.poisson(lam=mean, size=1)[0])

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
        return int(self.sample_n(customer_class, price, 1)[0])

    def sample_n(self, customer_class: CustomerClass, price: float, n: int):
        return self.rng.binomial(n, self.mean(customer_class=customer_class, price=price))

    @staticmethod
    def mean(customer_class: CustomerClass, price: float):
        return sigmoid(-price, -customer_class.crCenter, customer_class.sigmoidZ)


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
        return self.alpha / (self.alpha + self.beta)
