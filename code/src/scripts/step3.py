from src.Environment import Environment
from src.bandit.LearningStats import plot_results
from src.bandit.learner.ts.TSOptimalPriceLearner import TSOptimalPriceLearner
from src.algorithms import step1, expected_profit
from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.learner.ucb.UCBOptimalPriceLearner import UCBOptimalPriceLearner
import numpy as np


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    env = Environment()
    print(f'Running with seed {env.get_seed()}')
    env.print_summary()

    n_rounds = 365
    future_visits_delay = 30

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = PriceBanditEnvironment(env, prices, opt_bid, future_visits_delay)

    expected_profits = np.array([expected_profit(env, p, opt_bid) for p in prices])
    suboptimality_gaps = np.max(expected_profits) - expected_profits
    clairvoyant_cumulative_profits = np.cumsum([np.max(expected_profits)] * n_rounds)

    # plot_everything(env)

    ts_learner = TSOptimalPriceLearner(bandit_env)
    ucb_leaner = UCBOptimalPriceLearner(bandit_env)  # , lambda r: r % 50 == 0)

    ucb_leaner.learn(n_rounds)
    ucb_cumulative_profits = ucb_leaner.compute_cumulative_exp_profits(expected_profits)

    bandit_env.reset_state()

    ts_learner.learn(n_rounds)
    ts_cumulative_profits = ts_learner.compute_cumulative_exp_profits(expected_profits)

    optimal_profit = bandit_env.get_clairvoyant_optimal_expected_profit_not_discriminating()

    print(f"Optimal price is {opt_price} with expected profit {optimal_profit:.2f}")

    print("Price:             " + " ".join(f'{p:10d}' for p in bandit_env.prices))
    print("Expected profits:  " + " ".join(f'{e:10.2f}' for e in expected_profits))
    print("Gap:               " + " ".join(f'{e:10.2f}' for e in suboptimality_gaps))
    print()

    for name, learner in [("ucb", ucb_leaner), ("ts", ts_learner)]:
        print(name)
        print("Price:             " + " ".join(f'{p:10d}' for p in bandit_env.prices))
        print("Projected profits: " + " ".join(f'{p:10.2f}' for p in learner.compute_projected_profits()))
        print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in learner.compute_expected_profits()))
        print("Average crs:       " + " ".join(f'{p:10.2f}' for p in learner.get_average_conversion_rates()))
        print("Upper crs:         " + " ".join(f'{p:10.2f}' for p in learner.compute_projection_conversion_rates()))
        print("Number of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_number_of_pulls()))
        print()

    plot_results(["ucb", "ts"],
                 [ucb_cumulative_profits, ts_cumulative_profits],
                 clairvoyant_cumulative_profits, n_rounds)

    samples = 3000
    ucb_best = prices[np.argmax(ucb_leaner.compute_expected_profits())]
    ts_best = prices[np.argmax(ts_learner.compute_expected_profits())]
    target_prices = list({opt_price, ucb_best, ts_best})

    if ucb_best != opt_price:
        print('ucb Learned the wrong price!')

    if ts_best != opt_price:
        print('ts Learned the wrong price!')

    if len(target_prices) == 1:
        print('All the learners learned the right price')

    for target_price in target_prices:
        pricing_strategy = {c: target_price for c in env.get_features_combinations()}
        cum_profit = 0
        for __ in range(samples):
            _, _, _, _, _, profit = env.simulate_one_day_fixed_bid(pricing_strategy, opt_bid)
            cum_profit += profit
        print(f"Empiric profit of price {target_price} is {(cum_profit / samples):.2f}")


if __name__ == "__main__":
    main()
