from unittest import TestCase

import numpy as np

from src.Environment import Environment
from src.algorithms import step1


class Test(TestCase):
    def test_expected_profit_no_discrimination(self):
        samples = 10000
        tests = 10

        prices = np.arange(10, 101, 10)
        bids = np.arange(1, 100)

        cum_error = 0
        for test_n in range(1, tests+1):
            print(f'Running test n°{test_n:2d}/{tests}')
            env = Environment()
            opt_price, opt_bid, exp_profit = step1(env, prices, bids)

            pricing_strategy = {c:opt_price for c in env.get_features_combinations()}

            cum_profit = 0
            for __ in range(samples):
                _, _, _, _, _, profit = env.simulate_one_day_fixed_bid(pricing_strategy, opt_bid)
                cum_profit += profit

            average_profit = cum_profit / samples

            error = average_profit-exp_profit

            cum_error += error
            print(f'Run test n°{test_n:2d}/{tests} with error {error:.2f}')
        average_error = cum_error/tests

        print(f'Final average error: {average_error:.2f}')
        if abs(average_error) > 10:
            print(f'Failed with error {average_error:.2f} after {tests} tests with {samples} samples')
            self.fail()
