from code.src.bandit.BanditEnvironment import BanditEnvironment
from code.src.utils import NormalGamma
import numpy as np


class TSLearner:
    def __init__(self, n_arms: int, env: BanditEnvironment):
        self.n_arms = n_arms
        self.env = env
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.prices = []
        self.bids = []
        self.partial_rewards = [0]

        # In our model the likelihood is a gaussian function with unknown mean and variance, thus
        # the prior we consider is a normal gamma distribution, characterized by three parameters:
        # mu (mean), v, alpha (shape), beta (inverse scale)

        self.priors = [NormalGamma(1, 1, 1, 1) for i in range(n_arms)]

    def init_priors(self):
        rwds = []

        for i in range(self.n_arms):
            rwds.append(self.pull_arm_price(i, init=True))

        mu_bar = np.mean(rwds)

        for i in range(self.n_arms):
            self.priors[i].mu = mu_bar

    def pull_arm_price(self, a, init=False):
        # Get the reward from the environment
        rwd = self.env.round_bids_known(0, self.prices[a], self.bids[0])
        self.rewards_per_arm[a].append(rwd)
        self.partial_rewards.append(self.partial_rewards[-1] + rwd)

        if init:
            return rwd
        else:
            # Update the arm prior
            x_bar = np.mean(self.rewards_per_arm[a])
            mu0 = self.priors[a].mu
            v = self.priors[a].v
            alpha = self.priors[a].alpha
            beta = self.priors[a].beta
            n = len(self.rewards_per_arm[a])

            new_mu = (v * mu0 + n * x_bar) / (v + n)
            new_v = v + n
            new_alpha = alpha + (n / 2)
            new_beta = beta + np.sum([(x - x_bar) ** 2 for x in self.rewards_per_arm[a]]) + (n * v) / (v + n) * (
                    ((x_bar - mu0) ** 2) / 2)

            self.priors[a].update_params(new_mu, new_v, new_alpha, new_beta)

    def choose_arm(self):
        # Take a sample from each prior
        samples = [self.priors[a].sample() for a in range(self.n_arms)]
        # Return the best sample
        return np.argmax(samples)

    def learn_price(self, n_rounds, prices, bid):
        # Setup variables
        self.prices = prices
        self.bids = [bid]

        self.init_priors()

        # Go through all rounds
        for t in range(self.n_arms, n_rounds):
            a = self.choose_arm()
            self.pull_arm_price(a)

        self.pulled_arms_recap()

    def pulled_arms_recap(self):
        print("\nThompson Sampling Learner")

        for a in range(self.n_arms):
            print(
                f"[info] Arm {a} pulled {len(self.rewards_per_arm[a])} times - avg reward: {np.mean(self.rewards_per_arm[a])}")
