from unittest import TestCase

import numpy as np

from src.Environment import Environment
from src.algorithms import step1
from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.TSOptimalPriceLearner import TSOptimalPriceLearner


class TestRepeatable(TestCase):
    def test_repeatable(self):

        n_runs = 100
        n_rounds = 100
        future_visits_delay = 30

        prices = np.arange(10, 101, 10)
        bids = np.arange(1, 100)

        env = Environment()
        seed = env.get_seed()

        opt_price, opt_bid, profit = step1(env, prices, bids)
        bandit_env = PriceBanditEnvironment(env, prices, opt_bid, future_visits_delay)
        ts_learner = TSOptimalPriceLearner(bandit_env)

        ts_learner.learn(n_rounds)
        average_crs = ts_learner.get_average_conversion_rates()

        for i in range(1, n_runs + 1):
            env = Environment(random_seed=seed)

            opt_price, opt_bid, profit = step1(env, prices, bids)
            bandit_env = PriceBanditEnvironment(env, prices, opt_bid, future_visits_delay)

            ts_learner = TSOptimalPriceLearner(bandit_env)

            ts_learner.learn(n_rounds)

            av_crs = ts_learner.get_average_conversion_rates()

            if not np.equal(average_crs, av_crs).all():
                print(f'Failed at run n° {i} with seed {seed}')
                self.fail()

