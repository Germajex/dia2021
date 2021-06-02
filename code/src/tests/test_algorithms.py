from unittest import TestCase

import numpy as np

from src.Environment import Environment
from src.algorithms import expected_profit, step1


class TestStep1(TestCase):
    def test_step1_bruteforce(self):
        prices = np.linspace(1, 100, 100)
        bids = np.linspace(1, 100, 100)
        rng = np.random.default_rng()
        n_envs = 100

        for i in range(n_envs):
            print(f'Beginning step {i+1}/{n_envs}')
            # some seeds fail due to non unique optimal price (probably floating point rounding problems)
            # failing seed 1772628089

            seed = rng.integers(0, 2**32)
            env = Environment(seed)

            bf_p, bf_b, bf_profit = (prices[0], bids[0], expected_profit(env, prices[0], bids[0]))
            for p in prices:
                for b in bids:
                    profit = expected_profit(env, p, b)
                    if profit > bf_profit:
                        bf_p, bf_b, bf_profit = p, b, profit

            opt_p, opt_b, opt_profit = step1(env, prices, bids)

            if opt_p != bf_p or opt_b != bf_b:
                print(f'Failed with seed {env.get_seed()}')
                self.fail()

            print(f'Finished step {i+1}/{n_envs}')
