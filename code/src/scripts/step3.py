from src.Environment import Environment
from src.bandit.LearningStats import plot_results
from src.bandit.TSOptimalPriceLearner import TSOptimalPriceLearner
from src.algorithms import step1
from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.UCBOptimalPriceLearner import UCBOptimalPriceLearner
import numpy as np

from src.scripts.environment_plotter import plot_everything


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    env = Environment(2737403095)
    print(f'Running with seed {env.get_seed()}')
    n_rounds = 2000
    future_visits_delay = 30

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = BanditEnvironment(env, prices, opt_bid, future_visits_delay)
    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env.get_clairvoyant_cumulative_profits_not_discriminating(regret_length)

    #plot_everything(env)

    ts_learner = TSOptimalPriceLearner(bandit_env)
    ucb_leaner = UCBOptimalPriceLearner(bandit_env, lambda r: r % 50 == 51)

    ucb_leaner.learn(n_rounds)
    ucb_cumulative_profits = ucb_leaner.compute_cumulative_profits()

    bandit_env.reset_state()

    ts_learner.learn(n_rounds)
    ts_cumulative_profits = ts_learner.compute_cumulative_profits()

    optimal_profit = bandit_env.get_clairvoyant_optimal_expected_profit_not_discriminating()

    print(f"Optimal price is {opt_price} with expected profit {optimal_profit:.2f}")

    for name, learner in [("UCB", ucb_leaner), ("TS", ts_learner)]:
        print(name)
        print("Price:             " + " ".join(f'{p:10d}' for p in bandit_env.prices))
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in learner.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in learner.compute_expected_profits()))
        print("Averages:          " + " ".join(f'{p:10.2f}' for p in learner.get_average_conversion_rates()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_number_of_pulls()))
        print()

    plot_results(["UCB", "TS"], [ucb_cumulative_profits, ts_cumulative_profits], clairvoyant_cumulative_profits, regret_length, smooth=True)


if __name__ == "__main__":
    main()
