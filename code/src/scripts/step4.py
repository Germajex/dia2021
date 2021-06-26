from src.Environment import Environment
from src.bandit.TSOptimalPriceLearner import TSOptimalPriceLearner
from src.algorithms import step1
from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.UCBOptimalPriceDiscriminatingLearner import UCBOptimalPriceDiscriminatingLearner
from src.bandit.UCBOptimalPriceLearner import UCBOptimalPriceLearner
import numpy as np


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    env1 = Environment()
    env2 = Environment(env1.get_seed())

    print(f'Running with seed {env1.get_seed()}')
    n_rounds = 365
    future_visits_delay = 30

    opt_price_1, opt_bid_1, profit_1 = step1(env1, prices, bids)
    opt_price_2, opt_bid_2, profit_2 = step1(env2, prices, bids)

    bandit_env_1 = BanditEnvironment(env1, prices, opt_bid_1, future_visits_delay)
    bandit_env_2 = BanditEnvironment(env2, prices, opt_bid_2, future_visits_delay)

    ucb_learner = UCBOptimalPriceLearner(bandit_env_1)
    ucb_learner.learn(n_rounds)
    bandit_env_1.reset_state()

    print("UCB no discrimination")
    print("Projected profits: " + " ".join(f'{p:10.2f}' for p in ucb_learner.compute_projected_profits()))
    print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in ucb_learner.compute_expected_profits()))
    print("Averages:          " + " ".join(f'{p:10.2f}' for p in ucb_learner.get_average_conversion_rates()))
    print("Number of pulls:   " + " ".join(f'{p:10d}' for p in ucb_learner.get_number_of_pulls()))
    print()

    ucb_disc_leaner = UCBOptimalPriceDiscriminatingLearner(bandit_env_2)
    ucb_disc_leaner.learn(n_rounds)

    print("UCB with discrimination")
    for context in ucb_disc_leaner.get_contexts():
        print('Combinations of features:', *context.features)
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in ucb_disc_leaner.compute_context_projected_profit(context)))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in ucb_disc_leaner.compute_context_expected_profit(context)))
        print("Averages:          " + " ".join(f'{p:10.2f}' for p in ucb_disc_leaner.get_average_conversion_rates(context)))
        print("Number of pulls:   _____ come ?__________")
        print()


if __name__ == "__main__":
    main()
