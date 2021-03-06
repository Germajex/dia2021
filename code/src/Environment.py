import itertools
from collections import defaultdict

from numpy.random import default_rng, Generator

from src.constants import _Const
from src.CustomerClassCreator import CustomerClassCreator
from src.distributions import NewClicksDistribution, ClickConvertedDistribution, FutureVisitsDistribution, \
    CostPerClickDistribution


class Environment:
    def __init__(self, random_seed=None):
        CONST = _Const()

        if random_seed is None:
            self.rng: Generator = default_rng()
            random_seed = self.rng.integers(0, 2 ** 32)

        self._seed = random_seed
        self.rng: Generator = default_rng(seed=random_seed)

        self.combinations = list(itertools.product([True, False], repeat=2))
        self.feature_1_likelihood = self.rng.uniform(CONST.FEATURE_LIKELIHOOD_MIN, CONST.FEATURE_LIKELIHOOD_MAX)
        self.feature_2_likelihood = self.rng.uniform(CONST.FEATURE_LIKELIHOOD_MIN, CONST.FEATURE_LIKELIHOOD_MAX)

        self.likelihoods = {}
        for c in self.combinations:
            f1, f2 = c
            p1 = self.feature_1_likelihood if f1 else (1 - self.feature_1_likelihood)
            p2 = self.feature_2_likelihood if f2 else (1 - self.feature_2_likelihood)
            self.likelihoods[c] = p1 * p2

        self.classes = CustomerClassCreator().get_new_classes(self.rng,
                                                              self.combinations,
                                                              self.likelihoods,
                                                              CONST.N_CUSTOMER_CLASSES)

        self.average_tot_auctions = self.rng.uniform(CONST.AUCTIONS_MIN, CONST.AUCTIONS_MAX)
        self.item_base_price = self.rng.integers(CONST.BASE_PRICE_MIN, CONST.BASE_PRICE_MAX)
        self.newClicksC, self.newClicksZ = CustomerClassCreator().get_new_clicks_v_parameters(self.rng)

        self.distNewClicks = NewClicksDistribution(self.rng, self.newClicksC, self.newClicksZ,
                                                   self.average_tot_auctions,
                                                   self.likelihoods)
        self.distClickConverted = ClickConvertedDistribution(self.rng)
        self.distFutureVisits = FutureVisitsDistribution(self.rng)
        self.distCostPerClick = CostPerClickDistribution(self.rng)

        self.class_of_comb = {}
        for c in self.classes:
            for comb in c.features:
                self.class_of_comb[comb] = c

    def get_seed(self):
        return self._seed

    def margin(self, price: float):
        return price-self.item_base_price

    def print_summary(self):
        print(f'Theta 1: {self.feature_1_likelihood:.2f}')
        print(f'Theta 2: {self.feature_2_likelihood:.2f}')
        print(f'Average daily auctions: {self.average_tot_auctions:.2f}')
        print(f'50% winrate bid: {self.newClicksC:.2f}')
        print(f'Bid concentration: {self.newClicksZ:.2f}')
        print(f'Base price: {self.item_base_price:.2f}')
        print()
        print()
        self.print_class_summary()

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
        return self.combinations

    def get_features_comb_likelihood(self, f):
        return self.likelihoods[f]

    def simulate_one_day_fixed_bid(self, pricing_strategy, bid):
        bidding_strategy = defaultdict(lambda: bid)
        return self.simulate_one_day(pricing_strategy, bidding_strategy)

    def simulate_one_day_fixed_price(self, price, bidding_strategy):
        pricing_strategy = defaultdict(lambda: price)
        return self.simulate_one_day(pricing_strategy, bidding_strategy)

    def simulate_one_day_fixed_both(self, price, bid):
        pricing_strategy = defaultdict(lambda: price)
        bidding_strategy = defaultdict(lambda: bid)
        return self.simulate_one_day(pricing_strategy, bidding_strategy)

    # start simulation one day

    def simulate_one_day(self, pricing_strategy, bidding_strategy):
        purchases, tot_cost, new_future_visits, new_clicks = {}, {}, {}, {}

        auctions, new_clicks = self.distNewClicks.sample_bidding_strategy(bidding_strategy)
        profit = 0

        for c in self.classes:
            for comb in c.features:
                price = pricing_strategy[comb]
                bid = bidding_strategy[comb]
                purchases[comb] = self.distClickConverted.sample_n(c, price, new_clicks[comb])
                tot_cost[comb] = sum(self.distCostPerClick.sample_n(c, bid, new_clicks[comb]))
                new_future_visits[comb] = sum(self.distFutureVisits.sample_n(c, purchases[comb]))
                profit += self.margin(price) * (purchases[comb] + new_future_visits[comb]) - tot_cost[comb]

        return auctions, new_clicks, purchases, tot_cost, new_future_visits, profit

    # end simulation one day
