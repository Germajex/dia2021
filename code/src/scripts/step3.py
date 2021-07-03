from src.Environment import Environment
from src.bandit.LearningStats import plot_results
from src.bandit.TSOptimalPriceLearner import TSOptimalPriceLearner
from src.algorithms import step1, expected_profit
from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.UCBOptimalPriceLearner import UCBOptimalPriceLearner
import numpy as np


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    env = Environment()
    print(f'Running with seed {env.get_seed()}')
    env.print_summary()

    n_rounds = 2000
    future_visits_delay = 30

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = PriceBanditEnvironment(env, prices, opt_bid, future_visits_delay)
    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env.get_clairvoyant_cumulative_profits_not_discriminating(regret_length)
    expected_profits = np.array([expected_profit(env, p, opt_bid) for p in prices])
    suboptimality_gaps = np.max(expected_profits) - expected_profits
    clairvoyant_cumulative_profits_2 = np.cumsum([np.max(expected_profits)] * n_rounds)

    # plot_everything(env)

    ts_learner = TSOptimalPriceLearner(bandit_env)
    ucb_leaner = UCBOptimalPriceLearner(bandit_env)  # , lambda r: r % 50 == 0)

    ucb_leaner.learn(n_rounds)
    ucb_cumulative_profits = ucb_leaner.compute_cumulative_profits()
    ucb_cumulative_profits_2 = ucb_leaner.compute_cumulative_exp_profits(expected_profits)

    bandit_env.reset_state()

    ts_learner.learn(n_rounds)
    ts_cumulative_profits = ts_learner.compute_cumulative_profits()
    ts_cumulative_profits_2 = ts_learner.compute_cumulative_exp_profits(expected_profits)

    optimal_profit = bandit_env.get_clairvoyant_optimal_expected_profit_not_discriminating()

    print(f"Optimal price is {opt_price} with expected profit {optimal_profit:.2f}")

    print("Price:             " + " ".join(f'{p:10d}' for p in bandit_env.prices))
    print("Expected profits:  " + " ".join(f'{e:10.2f}' for e in expected_profits))
    print("Gap:               " + " ".join(f'{e:10.2f}' for e in suboptimality_gaps))
    print()

    for name, learner in [("UCB", ucb_leaner), ("TS", ts_learner)]:
        print(name)
        print("Price:             " + " ".join(f'{p:10d}' for p in bandit_env.prices))
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in learner.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in learner.compute_expected_profits()))
        print("Average crs:       " + " ".join(f'{p:10.2f}' for p in learner.get_average_conversion_rates()))
        print("Upper crs:         " + " ".join(f'{p:10.2f}' for p in learner.compute_projection_conversion_rates()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_number_of_pulls()))
        print()

    plot_results(["UCB", "TS"],
                 [ucb_cumulative_profits, ts_cumulative_profits],
                 clairvoyant_cumulative_profits, regret_length, smooth=True)

    plot_results(["UCB", "TS"],
                 [ucb_cumulative_profits_2, ts_cumulative_profits_2],
                 clairvoyant_cumulative_profits_2, n_rounds, smooth=True)

    samples = 3000
    ucb_best = prices[np.argmax(ucb_leaner.compute_expected_profits())]
    ts_best = prices[np.argmax(ts_learner.compute_expected_profits())]
    target_prices = list({opt_price, ucb_best, ts_best})

    for target_price in target_prices:
        pricing_strategy = {c: target_price for c in env.get_features_combinations()}
        cum_profit = 0
        for __ in range(samples):
            _, _, _, _, _, profit = env.simulate_one_day_fixed_bid(pricing_strategy, opt_bid)
            cum_profit += profit
        print(f"Empiric profit of price {target_price} is {(cum_profit / samples):.2f}")


if __name__ == "__main__":
    main()
