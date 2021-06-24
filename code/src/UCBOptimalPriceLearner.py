import numpy
import numpy as np

from src.OptimalPriceLearner import OptimalPriceLearner
from src.bandit.BanditEnvironment import BanditEnvironment
import matplotlib.pyplot as plt


class UCBOptimalPriceLearner(OptimalPriceLearner):
    def __init__(self, env: BanditEnvironment, show_round=lambda r:False):
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

    def get_average_conversion_rates(self):
        return self.compute_conversion_rates_averages()

    def compute_conversion_rates_radia(self):
        tot_clicks_per_arm = np.array([np.sum(self.new_clicks_per_arm[arm])
                                       for arm in range(self.n_arms)])

        return np.sqrt(2 * np.log(self.current_round) / tot_clicks_per_arm)

    def compute_conversion_rates_upper_bounds(self):
        averages = self.compute_conversion_rates_averages()
        radia = self.compute_conversion_rates_radia()
        upper_bounds = averages + radia

        return upper_bounds

    # Method used for debugging mainly. Plots the average reward and the confidence bounds used by the UCB algorithm
    def show_bounds(self):
        x = [a for a in range(self.n_arms)]

        fig, ax = plt.subplots(1)

        ax.set_title(f"Round {self.current_round}")

        ax.set_ylabel("expected cr")

        means = self.compute_conversion_rates_averages()

        ax.set_xlabel("arms")

        ax.set_xticklabels(self.prices)
        ax.set_xticks(x)

        ax.scatter(x, means)
        radia = self.compute_conversion_rates_radia()
        y_min = means-radia
        y_max = means+radia

        ax.vlines(x, y_min, y_max)
        ax.hlines([0, 1], -1, 11, colors='red')
        ax.set_xlim(-1, 11)

        plt.show()

    # Choose the arm to pull based on the reward upper bound (reward computed with cr upper bound).
    # Returns the index of the arm to be pulled
    def choose_arm(self, t, mode):
        # The choice of the arm to be pulled is based on the
        if mode == 'rwd':
            rwds_array = [self.ucb_upper_bound(a, t, mode) for a in range(self.n_arms)]
            return np.argmax(rwds_array)

        elif mode == 'cr':
            cr_upper_bound_array = [self.ucb_upper_bound(a, t, mode) for a in range(self.n_arms)]
            rwds_array = [self.env.get_total_reward(a, self.bids[0], cr) for a, cr in
                          zip(self.prices, cr_upper_bound_array)]
            return np.argmax(rwds_array)

    # Computes the ucb upper bound of a given arm at a given time
    def ucb_upper_bound(self, arm, time, mode):
        value_array = []
        if mode == 'cr':
            value_array = self.cr_per_arm
        elif mode == 'rwd':
            value_array = self.rewards_per_arm

        return np.mean(value_array[arm]) + self.ucb_confidence_bound(arm, time, mode)

    # Computes the ucb lower bound of a given arm at a given time
    def ucb_lower_bound(self, arm, time, mode):
        value_array = []
        if mode == 'cr':
            value_array = self.cr_per_arm
        elif mode == 'rwd':
            value_array = self.rewards_per_arm

        return np.mean(value_array[arm]) - self.ucb_confidence_bound(arm, time, mode)

    # Implementation of the UCB confidence bound. It needs the arm, the timestamp and an hyper-parameter c
    def ucb_confidence_bound(self, a, t, mode):
        if mode == 'cr':
            n_pulls = len(self.cr_per_arm[a])
            return self.c * np.sqrt(2 * np.log(t) / n_pulls)
        elif mode == 'rwd':
            n_pulls = len(self.rewards_per_arm[a])
            b = self.get_max_reward()
            sigma = 1/(t**4)
            num = self.c*np.log((4*self.n_arms*(t**3))/sigma)
            den = n_pulls

            return b*np.sqrt(num/den)

    def get_max_reward(self):
        max_rwd = 0
        for r in self.rewards_per_arm:
            r_max = np.max(r)
            if r_max > max_rwd:
                max_rwd = r_max

        return max_rwd

    # Pulls the given arm, which is a price to be tested. Also appends the collected reward, or the cr, depending on the
    # operation mode
    def pull_arm_price(self, a, mode):
        rwd, successes, failures = self.env.round_bids_fixed(self.prices[a], self.bids[0])

        self.cr_per_arm[a].append(successes / (successes + failures))
        self.rewards_per_arm[a].append(rwd)

        # We still want to record the cumulative rewards, to compute the regret
        self.history_rewards.append(rwd)

    # Prints out a brief recap of the activities done by the learner
    def pulled_arms_recap(self, mode):
        print(f"\nUCB Learner working in mode {mode}")

        for a in range(self.n_arms):
            n_pulls = len(self.rewards_per_arm[a])
            avg_reward = np.mean(self.rewards_per_arm[a])
            avg_cr = np.mean(self.cr_per_arm[a])

            print(
                f"[info] Arm {a} pulled {n_pulls} times - avg reward: {avg_reward:.2f} - avg cr: {avg_cr:.2f}")
