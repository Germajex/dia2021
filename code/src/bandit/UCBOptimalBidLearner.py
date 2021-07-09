import numpy as np
from matplotlib import pyplot as plt

from src.bandit.BidBanditEnvironment import BidBanditEnvironment
from src.bandit.OptimalBidLearner import OptimalBidLearner
from src.utils import max_ragged_matrix


class UCBOptimalBidLearner(OptimalBidLearner):
    def __init__(self, env: BidBanditEnvironment, show_round=lambda r: False):
        super().__init__(env)
        self.show_round = show_round

    def learn_one_round(self):
        super().learn_one_round()

        if self.show_round(self.current_round):
            self.show_bounds()

    def compute_projection_new_clicks(self):
        return self.compute_average_new_clicks() + self.compute_new_clicks_radia()

    def compute_new_clicks_radia(self):
        b = np.array([np.max(self.new_clicks_per_arm[arm]) for arm in range(self.n_arms)])

        return b * np.sqrt(2 * np.log(self.current_round) / self.get_number_of_pulls())

    def compute_average_new_clicks(self):
        average_clicks_per_arm = np.array([np.sum(np.array(self.new_clicks_per_arm[arm]) / len(self.new_clicks_per_arm[arm]))
                                           for arm in range(self.n_arms)])

        return average_clicks_per_arm

    # Method used for debugging mainly. Plots the average reward and the confidence bounds used by the UCB algorithm
    def show_bounds(self):
        x = [a for a in range(self.n_arms)]

        fig, ax = plt.subplots(1)

        ax.set_title(f"Round {self.current_round}")
        ax.set_ylabel("expected new_clicks")
        ax.set_xlabel("arms")
        ax.set_xticklabels(self.env.bids)
        ax.set_xticks(x)

        means = self.compute_average_new_clicks()
        radia = self.compute_new_clicks_radia()

        ax.errorbar(x, means, yerr=radia, color="black", capsize=5, fmt='o', markersize=4)

        b = max_ragged_matrix(self.new_clicks_per_arm)
        ax.hlines([0, b], -1, 11, colors='red')
        ax.set_xlim(-1, 11)

        print("Arm:             " + " ".join(f'{p:10d}' for p in range(self.n_arms)))
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in self.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in self.compute_expected_profits()))
        print("Averages:          " + " ".join(f'{p:10.2f}' for p in self.compute_average_new_clicks()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in self.get_number_of_pulls()))
        print()

        plt.show()
