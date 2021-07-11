from src.Environment import Environment
from src.bandit.LearningStats import plot_results
from src.bandit.TSOptimalPriceDiscriminatingLearner import TSOptimalPriceDiscriminatingLearner
from src.algorithms import step1, optimal_pricing_strategy_for_bid, expected_profit_of_pricing_strategy, \
    expected_profit_for_comb
from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.UCBOptimalPriceDiscriminatingLearner import UCBOptimalPriceDiscriminatingLearner
from src.bandit.UCBOptimalPriceLearner import UCBOptimalPriceLearner
import numpy as np


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    #      supponiamo che la ragione sia che le due combinazioni da splittare
    #      hanno prezzi ottimi molto diversi e in particolare una lo ha molto basso, così basso
    #      che non viene mai proposto e non il lower confidence bound non si stringe mai
    #      abbastanza per splittare

    env1 = Environment(3731638538)  # 2530307612)
    env2 = Environment(env1.get_seed())
    env3 = Environment(env1.get_seed())

    print(f'Running with seed {env1.get_seed()}')
    n_rounds = 800
    future_visits_delay = 30

    opt_price_1, opt_bid_1, profit_1 = step1(env1, prices, bids)
    opt_price_2, opt_bid_2, profit_2 = step1(env2, prices, bids)
    opt_price_3, opt_bid_3, profit_3 = step1(env3, prices, bids)

    bandit_env_1 = PriceBanditEnvironment(env1, prices, opt_bid_1, future_visits_delay)
    bandit_env_2 = PriceBanditEnvironment(env2, prices, opt_bid_2, future_visits_delay)
    bandit_env_3 = PriceBanditEnvironment(env3, prices, opt_bid_3, future_visits_delay)

    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env_1.get_clairvoyant_cumulative_profits_discriminating(regret_length)

    print(f'Optimal price without feature discrimination: {opt_price_1:.2f}, with profit {profit_1:.2f}')

    ucb_learner = UCBOptimalPriceLearner(bandit_env_1)
    ucb_learner.learn(n_rounds)
    bandit_env_1.reset_state()

    print("UCB no discrimination")
    print("Price:             " + " ".join(f'{p:10d}' for p in prices))
    print("\tProjected profits: " + " ".join(f'{p:10.2f}' for p in ucb_learner.compute_projected_profits()))
    print("\tExpected profits:  " + " ".join(f'{p:10.2f}' for p in ucb_learner.compute_expected_profits()))
    print("\tAverages:          " + " ".join(f'{p:10.2f}' for p in ucb_learner.get_average_conversion_rates()))
    print("\tNumber of pulls:   " + " ".join(f'{p:10d}' for p in ucb_learner.get_number_of_pulls()))
    print()

    opt_strategy = optimal_pricing_strategy_for_bid(env1, prices, opt_bid_1)
    opt_value = expected_profit_of_pricing_strategy(env1, opt_strategy, opt_bid_1)

    print("Optimal pricing strategy: " + ", ".join(
        f'{c.name}({", ".join(str(comb) for comb in c.features)}): {opt_strategy[c.features[0]]:.2f}'
        for c in env1.classes))
    print(f"With value: {opt_value:.2f}\n\n")
    clairvoyant_cumulative_profits2 = np.cumsum([opt_value] * n_rounds)
    expected_profits = {}
    for comb in env1.combinations:
        expected_profits[comb] = []
        for p in prices:
            expected_profits[comb].append(expected_profit_for_comb(env1, p, opt_bid_1, comb))

    print("Learning UCBDisc")

    ucb_disc_learner = UCBOptimalPriceDiscriminatingLearner(bandit_env_2)
    ucb_disc_learner.learn(n_rounds)
    #ucb_profits = ucb_disc_learner.compute_cumulative_profits()
    ucb_profits2 = ucb_disc_learner.compute_cumulative_exp_profits(expected_profits)
    bandit_env_2.reset_state()

    print("\nLearning TSDisc")
    ts_disc_learner = TSOptimalPriceDiscriminatingLearner(bandit_env_3)
    ts_disc_learner.learn(n_rounds)
    #ts_profits = ts_disc_learner.compute_cumulative_profits()
    ts_profits2 = ts_disc_learner.compute_cumulative_exp_profits(expected_profits)
    bandit_env_3.reset_state()

    for name, learner in [("UCB with discrimination", ucb_disc_learner), ("TS with discrimination", ts_disc_learner)]:
        print(name)
        for i, context in enumerate(learner.get_contexts(), start=1):
            print(f'Context n°{i} - Combinations of features:', *context.features)
            print("Price:             " + " ".join(f'{p:10d}' for p in prices))
            print("\tProjected profits: " + " ".join(
                f'{p:10.2f}' for p in learner.compute_context_projected_profit(context)))
            print("\tExpected profits:  " + " ".join(
                f'{p:10.2f}' for p in learner.compute_context_expected_profit(context)))
            print(
                "\tAverages:          " + " ".join(f'{p:10.2f}' for p in learner.get_average_conversion_rates(context)))
            print("\tNumber of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_context_number_of_pulls(context)))
            print()

    env1.print_summary()

    #plot_results(["UCB", "TS"], [ucb_profits, ts_profits], clairvoyant_cumulative_profits, regret_length)

    plot_results(["UCB", "TS"], [ucb_profits2, ts_profits2], clairvoyant_cumulative_profits2, n_rounds)


if __name__ == "__main__":
    main()
