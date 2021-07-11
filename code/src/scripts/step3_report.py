import os

from src.Environment import Environment
from src.bandit.LearningStats import plot_results
from src.bandit.TSOptimalPriceLearner import TSOptimalPriceLearner
from src.algorithms import step1, expected_profit
from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
import numpy as np

from src.bandit.UCBOptimalPriceLearner import UCBOptimalPriceLearner
from terminaltables import SingleTable, AsciiTable

from src.scripts.environment_plotter import plot_everything


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 101)
    interactive=False

    for envN, seedV in enumerate([3144548588, 1873674269]):

        env = Environment(random_seed=seedV) if seedV is not None else Environment()
        print(f'Running with seed {env.get_seed()}')

        n_rounds = 365
        future_visits_delay = 30

        opt_price, opt_bid, profit = step1(env, prices, bids)

        bandit_env = PriceBanditEnvironment(env, prices, opt_bid, future_visits_delay)

        ts_learner = TSOptimalPriceLearner(bandit_env)
        ucb_learner = UCBOptimalPriceLearner(bandit_env)

        ts_learner.learn(n_rounds)
        bandit_env.reset_state()
        ucb_learner.learn(n_rounds)

        expected_profits = np.array([expected_profit(env, p, opt_bid) for p in prices])
        gaps = np.max(expected_profits) - expected_profits
        ts_cumulative_profits = ts_learner.compute_cumulative_exp_profits(expected_profits)
        ucb_cumulative_profits = ucb_learner.compute_cumulative_exp_profits(expected_profits)
        clairvoyant_cumulative_profits = np.cumsum([np.max(expected_profits)] * n_rounds)

        if interactive:
            plot_results(["UCB", "TS"],
                         [ucb_cumulative_profits, ts_cumulative_profits],
                         clairvoyant_cumulative_profits, n_rounds)

        table_data = [['Price']]
        for p in prices:
            table_data.append([f'{p:.2f}'])

        table_data[0].append('Expected')
        for i, p in enumerate(prices):
            table_data[i + 1].append(f'{expected_profits[i]:.2f}')

        table_data[0].append('Gaps')
        for i, p in enumerate(prices):
            table_data[i + 1].append(f'{gaps[i]:.2f}')

        table_data[0] += ['UCB pulls', 'UCB expected', 'TS pulls', 'TS expected']
        for learner in [ucb_learner, ts_learner]:
            for i, (n, e) in enumerate(zip(learner.get_number_of_pulls(), learner.compute_expected_profits())):
                table_data[i + 1].append(f'{n}')
                table_data[i + 1].append(f'{e:.2f}')

        table = AsciiTable(table_data)
        for i in range(7):
            table.justify_columns[i] = 'right'
        print(table.table)

        dir = '../../../report/figures/step3'

        with open(dir+f'/output{envN}.txt', 'w', encoding='utf8') as output_file:
            output_file.write(f' Seed: {env.get_seed()}\n')
            output_file.write(table.table)

        plot_results(["UCB", "TS"],
                     [ucb_cumulative_profits, ts_cumulative_profits],
                     clairvoyant_cumulative_profits, n_rounds,
                     dest_file_path=dir+f'/plot{envN}')

        plot_everything(environment=env, path=dir+f'/env{envN}')


if __name__ == "__main__":
    main()
