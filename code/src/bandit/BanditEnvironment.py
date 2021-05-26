import numpy as np
from code.src.constants import _Const


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class BanditEnvironment:
    def __init__(self, n_arms, classes, bids=None, prices=None):
        self.n_arms = n_arms
        self.classes = classes
        self.bids = bids
        self.prices = prices

    # This methods return the best reward possible
    def get_optimal_reward(self, bid):
        CONST = _Const()

        prices = np.linspace(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX, 300)

        return np.max([np.sum([c.getRevenue(p, bid) for c in self.classes]) for p in prices])

    # Runs a round for a generic bandit. The learner must provide the price of the corresponding arm to be pulled,
    # this method returns the reward
    def round_bids_fixed(self, pulled_arm_val, bid):
        total_reward = 0
        for c in self.classes:
            # First extract the cr
            n_times = c.getNewClicks(bid)
            cr = np.random.binomial(n_times, c.getConversionRate(pulled_arm_val)) / n_times  # avg cr

            # get the reward
            class_n = self.classes.index(c)
            rwd = self.get_reward(class_n, pulled_arm_val, bid, cr)
            # sum to the total
            total_reward += rwd

        return total_reward

    # Given a class, price and bid, returns the revenue
    def get_reward(self, class_n, price, bid, cr):
        return self.classes[class_n].getRevenue(price, bid, cr=cr)

    # Returns a list containing all the cumulative rewards of the clairvoyant. Used mainly to compute regret
    def get_clairvoyant_partial_rewards(self, n_rounds):
        clairvoyant = [self.get_optimal_reward(self.bids[0]) * t for t in
                       range(n_rounds + 1)]
        return clairvoyant

    # Given a cumulative reward list, returns a list containing the cumulative regret
    def get_cumulative_regret(self, rewards):
        n_rounds = len(rewards) - 1
        return np.array(self.get_clairvoyant_partial_rewards(n_rounds) - np.array(rewards))
