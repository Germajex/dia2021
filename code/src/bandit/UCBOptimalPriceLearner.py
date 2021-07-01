import numpy as np

from src.bandit.OptimalPriceLearner import OptimalPriceLearner
from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
import matplotlib.pyplot as plt


class UCBOptimalPriceLearner(OptimalPriceLearner):
    def __init__(self, env: PriceBanditEnvironment, show_round=lambda r: False):
        super().__init__(env)
        self.show_round = show_round

    def learn_one_round(self):
        arm = self.choose_next_arm()
        self.pull_from_env(arm=arm)

        if self.show_round(self.current_round):
            self.show_bounds()

    def compute_projection_conversion_rates(self):
        return self.compute_conversion_rates_upper_bounds()

    def compute_conversion_rates_averages(self):
        return np.array([sum(self.purchases_per_arm[arm]) / sum(self.new_clicks_per_arm[arm])
                         for arm in range(self.n_arms)]).flatten()

    def compute_conversion_rates_radia(self):
        tot_clicks_per_arm = np.array([np.sum(self.new_clicks_per_arm[arm])
                                       for arm in range(self.n_arms)])

        return np.sqrt(2 * np.log(self.current_round) / tot_clicks_per_arm)

    def compute_conversion_rates_upper_bounds(self):
        averages = self.compute_conversion_rates_averages()
        radia = self.compute_conversion_rates_radia()
        upper_bounds = averages + radia

        return upper_bounds

    def get_average_conversion_rates(self):
        return self.compute_conversion_rates_averages()

    # Method used for debugging mainly. Plots the average reward and the confidence bounds used by the UCB algorithm
    def show_bounds(self):
        x = [a for a in range(self.n_arms)]

        fig, ax = plt.subplots(1)

        ax.set_title(f"Round {self.current_round}")
        ax.set_ylabel("expected cr")
        ax.set_xlabel("arms")
        ax.set_xticklabels(self.env.prices)
        ax.set_xticks(x)

        means = self.compute_conversion_rates_averages()
        radia = self.compute_conversion_rates_radia()

        ax.errorbar(x, means, yerr=radia, color="black", capsize=5, fmt='o', markersize=4)

        ax.hlines([0, 1], -1, 11, colors='red')
        ax.set_xlim(-1, 11)

        plt.show()
