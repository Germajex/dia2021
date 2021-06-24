import numpy as np

from src.OptimalPriceLearner import OptimalPriceLearner
from src.bandit.BanditEnvironment import BanditEnvironment
from src.utils import Beta


class TSOptimalPriceLearner(OptimalPriceLearner):
    def __init__(self, env: BanditEnvironment):
        super().__init__(env)
        self.beta_cr_priors = [Beta(1, 1, self.env.env.rng) for i in range(self.n_arms)]

    def compute_projection_conversion_rates(self):
        return self.sample_from_betas()

    def pull_from_env(self, arm: int):
        new_clicks, purchases, _, _ = super().pull_from_env(arm)
        self.update_betas(arm, purchases, new_clicks - purchases)

    def update_betas(self, arm: int, successes: int, failures: int):
        self.beta_cr_priors[arm].update_params(successes, failures)

    def sample_from_betas(self):
        sampled_crs = np.array([b.sample() for b in self.beta_cr_priors]).flatten()
        return sampled_crs

    def get_average_conversion_rates(self):
        averages = [b.mean() for b in self.beta_cr_priors]
        return averages