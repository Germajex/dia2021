import numpy as np
from matplotlib import pyplot as plt

from src.bandit.banditEnvironments.BidBanditEnvironment import BidBanditEnvironment
from src.bandit.learner.OptimalBidLearner import OptimalBidLearner


class UCBOptimalBidLearner(OptimalBidLearner):
    def __init__(self, env: BidBanditEnvironment, show_round=lambda r: False):
        super().__init__(env)
        self.show_round = show_round

    def learn_one_round(self):
        super().learn_one_round()

        if self.show_round(self.current_round):
            self.show_bounds()

    # start ucb bid win prob
    def compute_projection_auction_winning_probability_per_arm(self):
        avg = self.compute_average_auction_winning_probability_per_arm()
        radii = self.compute_auction_winning_probability_radii()
        upper_bounds = avg + radii

        return upper_bounds

    def compute_auction_winning_probability_radii(self):
        auctions_per_arm = np.array([sum(self.auctions_per_arm[a])
                                     for a in range(self.n_arms)])
        return np.sqrt(2 * np.log(self.current_round) / auctions_per_arm)

    # end ucb bid win prob

    # Method used for debugging mainly. Plots the average reward and the confidence bounds used by the ucb algorithm
    def show_bounds(self):
        x = [a for a in range(self.n_arms)]

        fig, ax = plt.subplots(1)

        ax.set_title(f"Round {self.current_round}")
        ax.set_ylabel("expected new_clicks")
        ax.set_xlabel("arms")
        ax.set_xticklabels(self.env.bids)
        ax.set_xticks(x)

        means = self.compute_average_auction_winning_probability_per_arm()
        radia = self.compute_auction_winning_probability_radii()

        ax.errorbar(x, means, yerr=radia, color="black", capsize=5, fmt='o', markersize=4)

        ax.hlines([0, 1], -1, 11, colors='red')
        ax.set_xlim(-1, 11)

        print("Arm:             " + " ".join(f'{p:10d}' for p in range(self.n_arms)))
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in self.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in self.compute_expected_profits()))
        print("Averages:          " + " ".join(f'{p:10.2f}' for p in self.compute_average_new_clicks_per_arm()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in self.get_number_of_pulls()))
        print()

        plt.show()
