import itertools

from numpy.random import default_rng, Generator

from src.constants import _Const
from src.CustomerClassCreator import CustomerClassCreator
from src.distributions import NewClicksDistribution, ClickConvertedDistribution, FutureVisitsDistribution, \
    CostPerClickDistribution, TotalAuctionsDistribution, AuctionsPerCombinationDistribution


class Environment:
    def __init__(self, random_seed=None):
        CONST = _Const()

        if random_seed is None:
            self.rng: Generator = default_rng()
            random_seed = self.rng.integers(0, 2 ** 32)

        self._seed = random_seed
        self.rng: Generator = default_rng(seed=random_seed)

        self.classes = CustomerClassCreator().get_new_classes(self.rng, CONST.N_CUSTOMER_CLASSES)
        self.feature_1_likelihood = self.rng.uniform(CONST.FEATURE_LIKELIHOOD_MIN, CONST.FEATURE_LIKELIHOOD_MAX)
        self.feature_2_likelihood = self.rng.uniform(CONST.FEATURE_LIKELIHOOD_MIN, CONST.FEATURE_LIKELIHOOD_MAX)

        self.average_tot_auctions = self.rng.uniform(CONST.AUCTIONS_MIN, CONST.AUCTIONS_MAX)

        self.newClicksC, self.newClicksZ = CustomerClassCreator().get_new_clicks_v_parameters(self.rng)

        likelihood_per_comb = {c: self.get_features_comb_likelihood(c) for c in self.get_features_combinations()}

        self.distTotalAuctions = TotalAuctionsDistribution(self.rng, self.average_tot_auctions)
        self.distAuctionsPerCombination = AuctionsPerCombinationDistribution(self.rng, likelihood_per_comb)
        self.distNewClicks = NewClicksDistribution(self.rng, self.newClicksC, self.newClicksZ, self.average_tot_auctions,
                                                   likelihood_per_comb)
        self.distClickConverted = ClickConvertedDistribution(self.rng)
        self.distFutureVisits = FutureVisitsDistribution(self.rng)
        self.distCostPerClick = CostPerClickDistribution(self.rng)

    def get_seed(self):
        return self._seed

    @staticmethod
    def margin(price: float):
        return price

    def print_class_summary(self):
        for c in self.classes:
            c.print_summary()
            print()

    def get_dist_new_clicks(self):
        return self.distNewClicks

    def get_dist_future_visits(self):
        return self.distFutureVisits

    def get_dist_click_converted(self):
        return self.distClickConverted

    def get_classes(self):
        return self.classes

    def get_features_combinations(self):
        return list(itertools.chain.from_iterable(c.features for c in self.classes))

    def get_features_comb_likelihood(self, f):
        f1 = self.feature_1_likelihood if f[0] else 1 - self.feature_1_likelihood
        f2 = self.feature_2_likelihood if f[1] else 1 - self.feature_2_likelihood
        return f1 * f2

    def simulate_one_day(self, pricing_strategy, bid):
        purchases, tot_cost_per_clicks, new_future_visits = {}, {}, {}
        tot_auctions = self.distTotalAuctions.sample()
        auctions = self.distAuctionsPerCombination.sample(tot_auctions)
        new_clicks = self.distNewClicks.sample(auctions, bid)

        for c in self.classes:
            for comb in c.features:
                price = pricing_strategy[comb]
                clicks = new_clicks[comb]
                purchases[comb] = self.distClickConverted.sample_n(c, price, clicks)
                tot_cost_per_clicks[comb] = sum(self.distCostPerClick.sample_n(c, bid, clicks))
                new_future_visits[comb] = sum(self.distFutureVisits.sample_n(c, purchases[comb]))

        return auctions, new_clicks, purchases, tot_cost_per_clicks, new_future_visits
