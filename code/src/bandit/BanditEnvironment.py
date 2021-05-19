import numpy as np
from code.src.constants import _Const

class BanditEnvironment:
    def __init__(self, n_arms, classes, bids=None, prices=None):
        self.n_arms = n_arms
        self.classes = classes
        self.bids = bids
        self.prices = prices

    def get_optimal_reward(self, class_n, bid):
        CONST = _Const()

        prices = np.linspace(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX, 300)

        return np.max([self.classes[class_n].getRevenue(p, bid) for p in prices])

    def round_bids_known(self, class_n, pulled_arm_val, bid):
        # First extract the cr
        n_times = self.classes[class_n].getNewClicks(bid)
        cr = np.random.binomial(n_times, self.classes[class_n].getConversionRate(pulled_arm_val)) / n_times  # avg cr

        # get the reward
        rwd = self.get_reward(class_n, pulled_arm_val, bid, cr)

        return rwd

    def get_reward(self, class_n, price, bid, cr):
        return self.classes[class_n].getRevenue(price, bid, cr=cr)
