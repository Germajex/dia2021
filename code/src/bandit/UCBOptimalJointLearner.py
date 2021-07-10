import numpy as np

from src.bandit.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.OptimalJointLearner import OptimalJointLearner
from src.utils import sum_ragged_matrix


class UCBOptimalJointLearner(OptimalJointLearner):
    def __init__(self, env: JointBanditEnvironment, show_round=lambda r: False):
        super().__init__(env)
        self.show_round = show_round

    def learn_one_round(self):
        super().learn_one_round()

    def compute_projection_conversion_rates(self):
        averages = self.compute_conversion_rates_averages()
        radia = self.compute_conversion_rates_radia()
        return averages + radia

    def compute_conversion_rates_averages(self):
        cr_averages = np.array([(np.sum(sum_ragged_matrix(self.purchases[p])) / np.sum(sum_ragged_matrix(self.new_clicks[p]))) if np.sum(sum_ragged_matrix(self.new_clicks[p])) else 0
                         for p in range(self.n_arms_price)])
        return cr_averages

    def compute_conversion_rates_radia(self):
        tot_clicks_per_arm = np.array([np.sum(sum_ragged_matrix(self.new_clicks[p]))
                                       for p in range(self.n_arms_price)])
        return np.sqrt(2 * np.log(self.current_round) / tot_clicks_per_arm)

    def compute_projection_auction_winning_probability(self, arm_price):
        averages = self.compute_auction_winning_probability_averages(self, arm_price)
        radia = self.compute_auction_winning_probability_radia()
        return averages + radia

    def compute_auction_winning_probability_averages(self, arm_price):
        return np.array([sum(self.new_clicks[arm_price][b]) / self.tot_auctions_per_bid[b] for b in range(self.n_arms_bid)])

    def compute_auction_winning_probability_radia(self):
        return np.sqrt(2 * np.log(self.current_round) / self.tot_auctions_per_bid)
