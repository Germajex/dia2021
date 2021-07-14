import numpy as np

from src.bandit.context.JointContext import JointContext
from src.utils import sum_ragged_matrix


class UCBJointContext(JointContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_projection_conversion_rates(self, current_round):
        averages = self.compute_conversion_rates_averages()
        radia = self.compute_conversion_rates_radia(current_round)
        return averages + radia

    def compute_conversion_rates_averages(self):
        cr_averages = np.array([(np.sum(sum_ragged_matrix(self.purchases[p])) / np.sum(
            sum_ragged_matrix(self.new_clicks[p]))) if np.sum(sum_ragged_matrix(self.new_clicks[p])) else 0
                                for p in range(self.n_arms_price)])
        return cr_averages

    def compute_conversion_rates_radia(self, current_round):
        tot_clicks_per_arm = np.array([np.sum(sum_ragged_matrix(self.new_clicks[p]))
                                       for p in range(self.n_arms_price)])
        np.where(tot_clicks_per_arm > 0, tot_clicks_per_arm, 1)

        return np.sqrt(2 * np.log(current_round) / tot_clicks_per_arm)

    def compute_projection_auction_winning_probability(self, current_round):
        averages = self.compute_auction_winning_probability_averages()
        radia = self.compute_auction_winning_probability_radia(current_round)
        return averages + radia

    def compute_auction_winning_probability_averages(self):
        new_c = [[] for b in range(self.n_arms_bid)]
        for b in range(self.n_arms_bid):
            for p in range(self.n_arms_price):
                new_c[b].extend(self.new_clicks[p][b])
        return np.array([sum(new_c[b]) / self.tot_auctions_per_bid[b] for b in range(self.n_arms_bid)]).flatten()

    def compute_auction_winning_probability_radia(self, current_round):
        return (np.sqrt(2 * np.log(current_round) / self.tot_auctions_per_bid)).flatten()
