import numpy as np
from numpy.random import Generator
from typing import List

from src.constants import _Const
from src.distributions import NewClicksDistribution, ClickConvertedDistribution, CostPerClickDistribution, \
    FutureVisitsDistribution
from src.CustomerClassCreator import CustomerClassCreator
from src.CustomerClass import CustomerClass

intlist = List[int]


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class BanditEnvironment:
    def __init__(self, n_arms: int, n_classes: int, generator: Generator):
        # Init local vars
        self.n_arms = n_arms
        self.classes = CustomerClassCreator().get_new_classes(generator, n_classes)
        self.n_classes = len(self.classes)
        self.rng = generator

        # Sample random values for distributions
        const = _Const()
        new_clicks_c = self.rng.choice(np.around(np.linspace(const.BID_MIN, const.BID_MAX, 20)))
        new_clicks_z = self.rng.choice(const.SIGMOID_Z_VALUES_NC)
        # Distribution of the new clicks
        self.new_clicks = NewClicksDistribution(generator, new_clicks_c, new_clicks_z)
        # Distribution for the conversion rate
        self.conversion_rates = ClickConvertedDistribution(generator)
        # Distribution for the cost per click
        self.cost_per_clicks = CostPerClickDistribution(generator)
        # Distribution for future visits
        self.future_visits = FutureVisitsDistribution(generator)

    # This methods returns the best reward possible, given a set of possible prices and a fixed bid
    def get_optimal_reward(self, prices, bid):
        optimal_arm = self.get_optimal_arm(prices, bid)
        rwd = np.sum([self.get_reward(class_n, prices[optimal_arm], bid, new_clicks_rounded=True)
                      for class_n in range(len(self.classes))])
        return rwd

    # This method returns the optimal CR, given a set of prices and a fixed bid
    def get_optimal_cr(self, prices, bid):
        arm = self.get_optimal_arm(prices, bid)

        total_clicks = np.sum([self.new_clicks.mean(c, bid) for c in self.classes])
        successes = np.sum(
            [self.conversion_rates.mean(c, prices[arm]) * self.new_clicks.mean(c, bid) for c in self.classes])

        cr = successes / total_clicks

        return cr

    # This method returns the index of the optimal arm (price). Given a set of prices and a fixed bid
    def get_optimal_arm(self, prices, bid):
        return np.argmax([np.sum([self.get_reward(c, p, bid) for c in range(self.n_classes)]) for p in prices])

    # Method used to sample a conversion rate from a customer class, given the price and the bid.
    # Returns the cr, the number of successes, the number of failures
    def sample_cr(self, class_n, price, bid):
        c = self.classes[class_n]
        new_clicks = int(self.new_clicks.mean(c, bid))
        successes = self.conversion_rates.sample_n(c, price, new_clicks)
        cr = successes / new_clicks
        failures = new_clicks - successes

        return cr, successes, failures

    # Runs a round for a generic bandit. The learner must provide the price of the corresponding arm to be pulled,
    # this method returns the reward and the success/failures of advertising
    def round_bids_fixed(self, pulled_arm_val, bid, features=None):
        total_reward = 0
        successes_total = 0
        new_clicks_total = 0

        for c in self.classes:
            if not self.class_matches_filters(features, c):
                print("WTFFF")
                continue

            class_n = self.classes.index(c)

            # First extract the cr
            cr, successes, failures = self.sample_cr(class_n, pulled_arm_val, bid)

            # get the reward
            rwd = self.get_reward(class_n, pulled_arm_val, bid, cr, new_clicks_rounded=True)

            # sum to the total
            total_reward += rwd
            successes_total += successes
            new_clicks_total += (successes + failures)

        return total_reward, successes_total, new_clicks_total - successes_total

    # Method used to check whether a class respects the feature filters provided
    def class_matches_filters(self, features, c: CustomerClass):
        if features is None:
            return True
        for f in c.features:
            res = True
            for i in range(len(features)):
                if features[i] is not None and features[i] != f[i]:
                    res = False
            if res:
                return True
        return False

    # Given a class, price and bid, returns the revenue
    def get_reward(self, class_n, price, bid, cr=None, new_clicks_rounded=False):
        cust_class = self.classes[class_n]

        if new_clicks_rounded:
            n = np.ceil(self.new_clicks.mean(cust_class, bid))
        else:
            n = self.new_clicks.mean(cust_class, bid)

        if cr is None:
            cr = self.conversion_rates.mean(cust_class, price)
        future_visits = self.future_visits.mean(cust_class)
        cost = self.cost_per_clicks.mean(bid)

        rev = (-cost * n) + (n * cr * price) + (n * future_visits * cr * price)

        return np.around(rev, 2)

    # Returns the total reward obtained by all classes, given a price, a bid and the cr
    def get_total_reward(self, price, bid, cr):
        return np.sum([self.get_reward(n, price, bid, cr) for n in range(len(self.classes))])

    # Returns the list of rewards obtained by a clairvoyant in the scenario of price learning, with fixed bid
    def get_clairvoyant_rewards_price(self, n_rounds, prices, bid):
        clairvoyant = [0]
        for t in range(n_rounds):
            optimal_reward = self.get_optimal_reward(prices, bid)
            clairvoyant.append(optimal_reward)
        return clairvoyant

    # Returns a list containing all the cumulative rewards of the clairvoyant in the scenario of price learning,
    # given a bid. Used mainly to compute regret.
    def get_clairvoyant_cumulative_rewards_price(self, n_rounds, prices, bid):
        clairvoyant = self.get_clairvoyant_rewards_price(n_rounds, prices, bid)
        return np.cumsum(clairvoyant)

    # Given a cumulative reward list, returns a list containing the cumulative regret. It's for pricing problem
    def get_cumulative_regret_price(self, prices, bid, rewards):
        n_rounds = len(rewards) - 1
        return np.array(self.get_clairvoyant_cumulative_rewards_price(n_rounds, prices, bid) - np.array(rewards))
