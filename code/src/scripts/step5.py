import numpy as np

from src.Environment import Environment
from src.algorithms import step1
from src.bandit.BidBanditEnvironment import BidBanditEnvironment
from src.bandit.LearningStats import plot_results
from src.bandit.UCBOptimalBidLearner import UCBOptimalBidLearner


def main():
    prices = np.arange(10, 101, 10)
    bids = np.linspace(1, 80, num=10, dtype=np.int64)

    env = Environment()
    print(f'Running with seed {env.get_seed()}')
    n_rounds = 4000
    future_visits_delay = 30

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = BidBanditEnvironment(env, opt_price, bids, future_visits_delay)
    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env.get_clairvoyant_cumulative_profits_not_discriminating(regret_length)

    ucb_leaner = UCBOptimalBidLearner(bandit_env, lambda r: r % 400 == 401)

    ucb_leaner.learn(n_rounds)
    ucb_cumulative_profits = bandit_env.get_learner_cumulative_profit_not_discriminating(ucb_leaner.pulled_arms)

    optimal_profit = bandit_env.get_clairvoyant_optimal_expected_profit_not_discriminating()

    print(f"Optimal bid is {opt_bid} with expected profit {optimal_profit:.2f}")

    for name, learner in [("UCB", ucb_leaner)]:
        print(name)
        print("Bid:               " + " ".join(f'{p:10d}' for p in bandit_env.bids))
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in learner.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in learner.compute_expected_profits()))
        print("Averages:          " + " ".join(f'{p:10.2f}' for p in learner.compute_average_new_clicks_per_arm()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_number_of_pulls()))
        print()

    env.print_summary()

    plot_results(["UCB"], [ucb_cumulative_profits], clairvoyant_cumulative_profits, regret_length, smooth=False)


if __name__ == "__main__":
    main()
