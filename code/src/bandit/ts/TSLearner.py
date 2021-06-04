from src.bandit.BanditEnvironment import BanditEnvironment
from src.utils import NormalGamma, Beta
import numpy as np


# This class represents a TS Learner
class TSLearner:
    def __init__(self, n_arms: int, rho: float, env: BanditEnvironment):
        self.n_arms = n_arms
        self.env = env
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.cr_per_arm = [[] for i in range(n_arms)]
        self.prices = []
        self.bids = []
        self.history_rewards = [0]
        self.rho = rho

        # For a Gaussian TS the likelihood is a gaussian function with unknown mean and variance, thus
        # the prior we consider is a normal gamma distribution, characterized by four parameters:
        # mu (mean), v, alpha (shape), beta (inverse scale)

        self.gaussian_price_priors = [NormalGamma(0, 0, 0.5, 0.5) for i in range(n_arms)]

        # In case of a standard TS, we will use a Beta distribution as prior. With alpha the number of successes
        # and beta the number of failures. In this case we will learn the cr associated with every arm and then
        # make decisions based on the expected reward given the cr

        self.beta_price_priors = [Beta(1, 1) for i in range(n_arms)]

    # This method pulls an arm a, where a is a specific price. This is used when the learner wants to learn the best
    # price
    def pull_arm_price(self, a, mode):
        # Get the reward from the environment
        rwd, successes, failures = self.env.round_bids_fixed(self.prices[a], self.bids[0])
        self.rewards_per_arm[a].append(rwd)
        self.cr_per_arm[a].append(successes / (successes + failures))
        self.history_rewards.append(rwd)

        # Update the correct prior distribution according to the learning mode
        if mode == 'rwd':
            self.update_gaussian_price_priors(a, rwd)
        elif mode == 'cr':
            self.update_beta_price_priors(a, successes, failures)

    def get_cumulative_rewards(self):
        return np.cumsum(self.history_rewards)

    # This method updates the prior distributions of the price arms, according to
    # Zhu, Tan (https://arxiv.org/pdf/2002.00232.pdf)
    def update_gaussian_price_priors(self, arm_n, sample):
        x = sample
        prior = self.gaussian_price_priors[arm_n]

        # utility local vars
        mu = prior.mu
        v = prior.v
        a = prior.alpha
        b = prior.beta

        # actual update
        prior.mu = (v / (v + 1)) * mu + (1 / (v + 1)) * x
        prior.v = v + 1
        prior.alpha = a + 0.5
        prior.beta = b + ((v / (v + 1)) * (((x - mu) ** 2) / 2))

    # This method updated the Beta prior distributions of the price arms. This is the standard update done by a
    # classic TS
    def update_beta_price_priors(self, arm_n, successes, failures):
        self.beta_price_priors[arm_n].update_params(successes, failures)

    # Method used to select the most promising price to be tested. Returns the index of the selected arm
    def choose_arm_price(self, mode):
        if mode == 'rwd':
            # According to Zhu, Tan (https://arxiv.org/pdf/2002.00232.pdf)
            samples = []
            for p in self.gaussian_price_priors:
                tau, theta = p.sample_zhu_tan()
                samples.append((self.rho * theta) - (1 / tau))

            return np.argmax(samples)

        elif mode == 'cr':
            # Standard TS with Beta prior
            samples = [self.beta_price_priors[b].sample() for b in range(self.n_arms)]  # sample a conversion rate
            # With the sampled cr, estimate the reward we would get, then select the best arm accordingly
            estimated_rewards = [self.env.get_total_reward(p, self.bids[0], sample) for p, sample in
                                 zip(self.prices, samples)]

            return np.argmax(estimated_rewards)

    # The 'core' Thompson Sampling algorithm to select the best price, given the number of rounds, the prices
    # and the fixed bid
    def learn_price(self, n_rounds, prices, bid, mode='cr', verbose=True):
        # Setup variables
        self.prices = prices
        self.bids = [bid]

        if mode == 'rwd':
            # Algorithm by Zhu, Tan (https://arxiv.org/pdf/2002.00232.pdf)

            # Starting round robin to initialize prior distributions
            for a in range(self.n_arms):
                self.pull_arm_price(a, mode)

            # Go through all rounds
            for t in range(self.n_arms, n_rounds):
                a = self.choose_arm_price(mode)
                self.pull_arm_price(a, mode)

        elif mode == 'cr':
            # There is no init phase here, the priors are all the same
            # we can go straight to sample -> evaluate -> pull
            for t in range(n_rounds):
                a = self.choose_arm_price(mode)
                self.pull_arm_price(a, mode)

        else:
            print(f"[error] !!! learning mode {mode} is not available !!!")

        if verbose:
            self.pulled_price_arms_recap(mode)

    # Prints out a brief recap of the activities done by the learner
    def pulled_price_arms_recap(self, mode):
        print(f"\nThompson Sampling Learner operating in {mode} mode, rho={self.rho}")

        for a in range(self.n_arms):
            if len(self.rewards_per_arm[a]) == 0:
                print(f"[info] Arm {a} was never pulled")
            else:
                n_pulls = len(self.rewards_per_arm[a])
                avg_rwd = np.mean(self.rewards_per_arm[a])
                avg_cr = np.mean(self.cr_per_arm[a])
                print(
                    f"[info] Arm {a} pulled {n_pulls} times - avg reward: {avg_rwd:.2f} - avg cr: {avg_cr:.2f}")
