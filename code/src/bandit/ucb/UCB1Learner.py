import numpy as np
from code.src.bandit.BanditEnvironment import BanditEnvironment
import matplotlib.pyplot as plt


class UCB1Learner:
    def __init__(self, n_arms: int, env: BanditEnvironment):
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.env = env
        self.n_arms = n_arms
        self.prices = []
        self.bids = []
        self.partial_rewards = [0]
        self.c = 500

    # This method runs a first round of exploration, initializing the UCB algorithm
    def explore_price(self):
        for a in range(self.n_arms):
            self.pull_arm_price(a)

    # The 'core' UCB algorithm, it learns the best price possible, given the number of rounds, the possible prices
    # and the fixed bid
    def learn_price(self, n_rounds, prices, bid, verbose=True):
        self.prices = prices
        self.bids = [bid]

        self.explore_price()
        # self.wait_and_show(self.n_arms)

        for i in range(self.n_arms, n_rounds):
            a = self.choose_arm(i)
            self.pull_arm_price(a)

        if verbose:
            self.pulled_arms_recap()

    # Method used for debugging mainly. Plots the average reward and the confidence bounds used by the UCB algorithm
    def wait_and_show(self, t):
        print(f"[info] Showing round {t}")

        x = [a for a in range(self.n_arms)]
        y = [np.mean(self.rewards_per_arm[a]) for a in range(self.n_arms)]

        fig, ax = plt.subplots(1)

        ax.set_xlabel("arms")
        ax.set_ylabel("expected rwd")

        ax.set_xticklabels(self.prices)
        ax.set_xticks(x)

        ax.scatter(x, y)

        y_min = [self.ucb_lower_bound(a, t) for a in range(self.n_arms)]
        y_max = [self.ucb_upper_bound(a, t) for a in range(self.n_arms)]

        ax.vlines(x, y_min, y_max)

        plt.show()

    # Choose the arm to pull based on the reward upper bound (reward computed with cr upper bound).
    # Returns the index of the arm to be pulled
    def choose_arm(self, t):
        rwds_array = [self.ucb_upper_bound(a, t) for a in range(self.n_arms)]
        return np.argmax(rwds_array)

    # Computes the ucb upper bound of a given arm at a given time
    def ucb_upper_bound(self, arm, time):
        return np.mean(self.rewards_per_arm[arm]) + self.ucb_confidence_bound(arm, time, self.c)

    # Computes the ucb lower bound of a given arm at a given time
    def ucb_lower_bound(self, arm, time):
        return np.mean(self.rewards_per_arm[arm]) - self.ucb_confidence_bound(arm, time, self.c)

    # Implementation of the UCB confidence bound. It needs the arm, the timestamp and an hyperparameter c
    def ucb_confidence_bound(self, a, t, c):
        return c * np.sqrt(2 * np.log(t) / len(self.rewards_per_arm[a]))

    # Pulls the given arm, which is a price to be tested. Also appends the collected reward.
    def pull_arm_price(self, a):
        rwd = self.env.round_bids_fixed(self.prices[a], self.bids[0])[0]

        self.rewards_per_arm[a].append(rwd)

        self.partial_rewards.append(self.partial_rewards[-1] + rwd)

    # Prints out a brief recap of the activities done by the learner
    def pulled_arms_recap(self):
        print("\nUCB Learner")

        for a in range(self.n_arms):
            print(
                f"[info] Arm {a} pulled {len(self.rewards_per_arm[a])} times - avg reward: {np.mean(self.rewards_per_arm[a])}")
