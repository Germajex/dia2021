import numpy as np
from terminaltables import AsciiTable

from src.Environment import Environment
from src.algorithms import step1, optimal_pricing_strategy_for_bid, expected_profit_of_pricing_strategy, \
    expected_profit_for_comb
from src.bandit.LearningStats import plot_results
from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.learner.ts.TSOptimalPriceDiscriminatingLearner import TSOptimalPriceDiscriminatingLearner
from src.bandit.learner.ucb.UCBOptimalPriceDiscriminatingLearner import UCBOptimalPriceDiscriminatingLearner
from src.scripts.environment_plotter import plot_everything


def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 101)
    interactive = False

    n_rounds = 365
    future_visits_delay = 30

    for envN, seedV in enumerate([3511939391, 1740212098, 4059059292]):

        env = Environment(random_seed=seedV) if seedV is not None else Environment()

        print(f'Running with seed {env.get_seed()}')

        opt_price, opt_bid, profit = step1(env, prices, bids)

        bandit_env = PriceBanditEnvironment(env, prices, opt_bid, future_visits_delay)

        opt_strategy = optimal_pricing_strategy_for_bid(env, prices, opt_bid)
        opt_value = expected_profit_of_pricing_strategy(env, opt_strategy, opt_bid)

        clairvoyant_cumulative_profits = np.cumsum([opt_value] * n_rounds)
        expected_profits = {comb: [expected_profit_for_comb(env, p, opt_bid, comb) for p in prices]
                            for comb in env.combinations}

        print("Learning UCBDisc")
        ucb_disc_learner = UCBOptimalPriceDiscriminatingLearner(bandit_env)
        ucb_disc_learner.learn(n_rounds)
        bandit_env.reset_state()
        ucb_profits = ucb_disc_learner.compute_cumulative_exp_profits(expected_profits)

        print("\nLearning TSDisc")
        ts_disc_learner = TSOptimalPriceDiscriminatingLearner(bandit_env)
        ts_disc_learner.learn(n_rounds)
        ts_profits = ts_disc_learner.compute_cumulative_exp_profits(expected_profits)
        bandit_env.reset_state()

        for name, learner in [("ucb with discrimination", ucb_disc_learner),
                              ("ts with discrimination", ts_disc_learner)]:
            print(name)
            for i, context in enumerate(learner.get_contexts(), start=1):
                print(f'context n°{i} - Combinations of features:', *context.features)
                print("Price:             " + " ".join(f'{p:10d}' for p in prices))
                print("\tProjected profits: " + " ".join(
                    f'{p:10.2f}' for p in learner.compute_context_projected_profit(context)))
                print("\tExpected profits:  " + " ".join(
                    f'{p:10.2f}' for p in learner.compute_context_expected_profit(context)))
                print(
                    "\tAverages:          " + " ".join(
                        f'{p:10.2f}' for p in learner.get_average_conversion_rates(context)))
                print("\tNumber of pulls:   " + " ".join(
                    f'{p:10d}' for p in learner.get_context_number_of_pulls(context)))
                print()

        env.print_summary()

        if interactive:
            plot_results(["ucb", "ts"],
                         [ucb_profits, ts_profits],
                         clairvoyant_cumulative_profits, n_rounds)

        optimal_pricing_str = "Optimal pricing strategy: " + ", ".join(
            f'{c.name}({", ".join(str(comb[0])[0]+str(comb[1])[0] for comb in c.features)}): {opt_strategy[c.features[0]]:.2f}'
            for c in env.classes)
        print(optimal_pricing_str)
        print(f"With value: {opt_value:.2f}\n\n")

        table_data_ucb, table_data_ts = [], []

        for table_data, learner in [(table_data_ucb, ucb_disc_learner), (table_data_ts, ts_disc_learner)]:
            table_data.append(['Price'])
            for p in prices:
                table_data.append([f'{p:.2f}'])

            for context in learner.get_contexts():
                context_str = ",".join(str(c[0])[0] + str(c[1])[0] for c in context.features)

                table_data[0].append(context_str)
                for i in range(len(prices)):
                    table_data[i + 1].append('')

                expected_profits_context = np.array([
                    sum(expected_profits[comb][i] for comb in context.features)
                    for i, price in enumerate(prices)
                ])
                gaps_context = np.max(expected_profits_context)-expected_profits_context
                norm_gaps_context = gaps_context / np.max(gaps_context)

                table_data[0].append('Gaps')
                for i, price in enumerate(prices):
                    table_data[i + 1].append(f'{norm_gaps_context[i]*100:4.1f}')

                table_data[0].append('Pulls')
                for i, n_pulls in enumerate(learner.get_context_number_of_pulls(context)):
                    table_data[i + 1].append(f'{n_pulls}')

        ucb_table = AsciiTable(table_data_ucb)
        for i in range(len(table_data_ucb[0])):
            ucb_table.justify_columns[i] = 'right'
        print(ucb_table.table)

        ts_table = AsciiTable(table_data_ts)
        for i in range(len(table_data_ts[0])):
            ts_table.justify_columns[i] = 'right'
        print(ts_table.table)

        dir = '../../../report/figures/step4'

        with open(dir + f'/output{envN}.txt', 'w', encoding='utf8') as output_file:
            output_file.write(f' Seed: {env.get_seed()}\n')
            output_file.write(' Legend: context, normalized gap, number of pulls\n')
            output_file.write(' UCB with context generation:\n')
            output_file.write(ucb_table.table)
            output_file.write('\n\n Performed splits:\n')
            for round_n, feature, incentive in ucb_disc_learner.get_perfomed_splits():
                output_file.write(f' Round{round_n:3d} split on feature {feature+1} '
                                  f'with incentive {incentive:.2f}\n')
            output_file.write('\n')
            output_file.write(' TS with context generation:\n')
            output_file.write(ts_table.table)
            output_file.write('\n\n Performed splits:\n')
            for round_n, feature, incentive in ts_disc_learner.get_perfomed_splits():
                output_file.write(f' Round{round_n:3d} split on feature {feature+1} '
                                  f'with incentive {incentive:.2f}\n')
            output_file.write("\n\n" + optimal_pricing_str)

        plot_results(["ucb", "ts"],
                     [ucb_profits, ts_profits],
                     clairvoyant_cumulative_profits, n_rounds,
                     dest_file_path=dir + f'/plot{envN}')

        plot_everything(environment=env, path=dir + f'/env{envN}')


if __name__ == "__main__":
    main()
