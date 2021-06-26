from src.Environment import Environment
from src.bandit.TSOptimalPriceLearner import TSOptimalPriceLearner
from src.algorithms import step1
from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.UCBOptimalPriceLearner import UCBOptimalPriceLearner
import numpy as np


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    env = Environment()
    print(f'Running with seed {env.get_seed()}')
    n_rounds = 365
    future_visits_delay = 30

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = BanditEnvironment(env, prices, opt_bid, future_visits_delay)

    ts_learner = TSOptimalPriceLearner(bandit_env)
    ucb_leaner = UCBOptimalPriceLearner(bandit_env, lambda r: r % 50 == 0)

    ucb_leaner.learn(n_rounds)

    bandit_env.reset_state()

    ts_learner.learn(n_rounds)

    for name, learner in [("UCB", ucb_leaner), ("TS", ts_learner)]:
        print(name)
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in learner.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in learner.compute_expected_profits()))
        print("Averages:          " + " ".join(f'{p:10.2f}' for p in learner.get_average_conversion_rates()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_number_of_pulls()))
        print()


if __name__ == "__main__":
    main()
