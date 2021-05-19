from code.src.bandit.BanditEnvironment import BanditEnvironment
from code.src.utils import NormalGamma
import numpy as np


class TSLearner:
    def __init__(self, n_arms: int, rho: float, env: BanditEnvironment):
        self.n_arms = n_arms
        self.env = env
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.prices = []
        self.bids = []
        self.partial_rewards = [0]
        self.rho = rho

        # In our model the likelihood is a gaussian function with unknown mean and variance, thus
        # the prior we consider is a normal gamma distribution, characterized by four parameters:
        # mu (mean), v, alpha (shape), beta (inverse scale)

        self.priors = [NormalGamma(0, 0, 0.5, 0.5) for i in range(n_arms)]

    def pull_arm_price(self, a):
        # Get the reward from the environment
        rwd = self.env.round_bids_known(0, self.prices[a], self.bids[0])
        self.rewards_per_arm[a].append(rwd)
        self.partial_rewards.append(self.partial_rewards[-1] + rwd)

        self.update_prior(a, rwd)

    def init_pull(self, a):
        # Get the reward from the environment
        rwd = self.env.round_bids_known(0, self.prices[a], self.bids[0])
        self.rewards_per_arm[a].append(rwd)
        self.partial_rewards.append(self.partial_rewards[-1] + rwd)

        pass

    def update_prior(self, arm_n, sample):
        x = sample
        prior = self.priors[arm_n]

        mu = prior.mu
        v = prior.v
        a = prior.alpha
        b = prior.beta

        prior.mu = (v / (v + 1)) * mu + (1 / (v + 1)) * x
        prior.v = v + 1
        prior.alpha = a + 0.5
        prior.beta = b + ((v / (v + 1)) * (((x - mu) ** 2) / 2))

    def choose_arm(self):
        # According to Zhu, Tan (https://arxiv.org/pdf/2002.00232.pdf)
        samples = []
        for p in self.priors:
            tau, theta = p.sample_zhu_tan()
            samples.append(self.rho * theta - 1 / tau)

        return np.argmax(samples)

    def learn_price(self, n_rounds, prices, bid):
        # Setup variables
        self.prices = prices
        self.bids = [bid]

        # Algorithm by Zhu, Tan (https://arxiv.org/pdf/2002.00232.pdf)

        # Starting round robin to initialize prior distributions
        for a in range(self.n_arms):
            self.pull_arm_price(a)

        # Go through all rounds
        for t in range(self.n_arms, n_rounds):
            a = self.choose_arm()
            self.pull_arm_price(a)

        self.pulled_arms_recap()

    def pulled_arms_recap(self):
        print(f"\nThompson Sampling Learner rho={self.rho}")

        for a in range(self.n_arms):
            print(
                f"[info] Arm {a} pulled {len(self.rewards_per_arm[a])} times - avg reward: {np.mean(self.rewards_per_arm[a])}")
