import numpy as np

from src.Environment import Environment
from src.algorithms import expected_profit
from unittest import TestCase


class TestStep1(TestCase):
    def test_unique_best_p(self):
        prices = np.linspace(1, 100, 100)
        bids = np.linspace(1, 100, 100)
        rng = np.random.default_rng()

        n_envs = 100

        # failing seed 1772628089 if round() is used in NewClicks.mean
        for i in range(n_envs):
            print(f'Beginning step {i + 1}/{n_envs}')
            seed = rng.integers(0, 2**32)
            env = Environment(seed)

            best_ps = [
                max(prices, key=lambda p1: expected_profit(env, p1, b))
                for b in bids
            ]

            opt_ps = np.unique(best_ps)

            if len(opt_ps) > 1:
                print(f'Failed with seed {env.get_seed()}')
                self.fail()

            print(f'Finished step {i + 1}/{n_envs}')

