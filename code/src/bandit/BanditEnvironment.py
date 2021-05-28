import numpy as np
from numpy.random import Generator
from typing import List

from code.src.CustomerClassCreator import CustomerClassCreator
from code.src.constants import _Const

intlist = List[int]


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class BanditEnvironment:
    def __init__(self, n_arms: int, n_classes: int, generator: Generator):
        self.n_arms = n_arms
        self.classes = CustomerClassCreator().getNewClasses(generator, n_classes)

    # This methods return the best reward possible, given a bid
    def get_optimal_reward(self, bid):
        CONST = _Const()

        prices = np.linspace(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX, 300)

        return np.max([np.sum([c.getRevenue(p, bid) for c in self.classes]) for p in prices])

    # Runs a round for a generic bandit. The learner must provide the price of the corresponding arm to be pulled,
    # this method returns the reward and the success/failures of advertising
    def round_bids_fixed(self, pulled_arm_val, bid):
        total_reward = 0
        successes_total = 0
        new_clicks_total = 0

        for c in self.classes:
            # First extract the cr
            n_clicks = c.getNewClicks(bid)
            successes = np.random.binomial(n_clicks, c.getConversionRate(pulled_arm_val))
            cr = successes / n_clicks  # avg cr

            # get the reward
            class_n = self.classes.index(c)
            rwd = self.get_reward(class_n, pulled_arm_val, bid, cr)

            # sum to the total
            total_reward += rwd
            successes_total += successes
            new_clicks_total += n_clicks

        return total_reward, successes_total, new_clicks_total - successes_total

    # Given a class, price and bid, returns the revenue
    def get_reward(self, class_n, price, bid, cr):
        return self.classes[class_n].getRevenue(price, bid, cr=cr)

    # Returns the total reward obtained by all classes, given a price, a bid and the cr
    def get_total_reward(self, price, bid, cr):
        return np.sum([self.get_reward(n, price, bid, cr) for n in range(len(self.classes))])

    # Returns a list containing all the cumulative rewards of the clairvoyant, given a bid.
    # Used mainly to compute regret, for the pricing problem
    def get_clairvoyant_partial_rewards_price(self, n_rounds, bid):
        clairvoyant = [self.get_optimal_reward(bid) * t for t in
                       range(n_rounds + 1)]
        return clairvoyant

    # Given a cumulative reward list, returns a list containing the cumulative regret. It's for pricing problem
    def get_cumulative_regret_price(self, bid, rewards):
        n_rounds = len(rewards) - 1
        return np.array(self.get_clairvoyant_partial_rewards_price(n_rounds, bid) - np.array(rewards))
