import numpy as np
from BanditEnviroment import BanditEnviroment
import matplotlib.pyplot as plt

class UCB1Learner:
    def __init__(self, n_arms : int, env : BanditEnviroment):
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.cr_per_arm = [[] for i in range(n_arms)]
        self.env = env
        self.n_arms = n_arms
        self.prices = []
        self.bids = []

    def cr_empirical_mean(self, arm):
        return np.mean(self.cr_per_arm[arm])

    def explore_price(self):
        for a in range(self.n_arms):
            self.pull_arm_price(a)

    def learn_price(self, n_rounds, prices, bid):
        self.prices = prices
        self.bids =[bid]

        self.explore_price()
        self.wait_and_show(self.n_arms)

        for i in range(self.n_arms, n_rounds):
            a = self.chooseArm(i)
            self.pull_arm_price(a)

            if i % 100 == 0:
                self.pulled_arms_recap()
                self.wait_and_show(i)

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

        y_min = [self.env.get_reward(0, self.prices[a], self.bids[0], self.cr_lower_bound(a, t)) for a in range(self.n_arms)]
        y_max = [self.env.get_reward(0, self.prices[a], self.bids[0], self.cr_upper_bound(a, t)) for a in range(self.n_arms)]

        ax.vlines(x, y_min, y_max)

        plt.show()

    # Choose the arm to pull based on the reward upper bound (reward computed with cr upper bound)
    def chooseArm(self, t):
        rwds_array = [self.env.get_reward(0, self.prices[a], self.bids[0], self.cr_upper_bound(a, t)) for a in range(self.n_arms)]

        return np.argmax(rwds_array)

    def cr_upper_bound(self, a, t):
        cr = self.cr_empirical_mean(a)+self.ucb1_confidence_bound_cr(a, t)
        return cr

    def cr_lower_bound(self, a, t):
        cr = self.cr_empirical_mean(a) - self.ucb1_confidence_bound_cr(a, t)
        return cr

    def ucb1_confidence_bound_cr(self, a, t):
        return np.sqrt(2*np.log(t)/len(self.rewards_per_arm[a]))

    def pull_arm_price(self, a):
        rwd, cr = self.env.round_bids_known(0, self.prices[a], self.bids[0])

        self.rewards_per_arm[a].append(rwd)
        self.cr_per_arm[a].append(cr)

    def pulled_arms_recap(self):
        for a in range(self.n_arms):
            print(f"[info] Arm {a} pulled {len(self.cr_per_arm[a])} times - avg reward: {np.mean(self.rewards_per_arm[a])}")