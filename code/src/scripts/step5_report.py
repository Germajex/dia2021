import numpy as np
from terminaltables import AsciiTable

from src.Environment import Environment
from src.algorithms import step1, expected_profit
from src.bandit.banditEnvironments.BidBanditEnvironment import BidBanditEnvironment
from src.bandit.LearningStats import plot_results
from src.bandit.learner.ucb.UCBOptimalBidLearner import UCBOptimalBidLearner
from src.constants import _Const
from src.scripts.environment_plotter import plot_everything


def main():
    CONST = _Const()

    interactive = False
    prices = np.arange(1, 101)
    bids = np.linspace(1, CONST.BID_MAX+10, num=10, dtype=np.int32)

    n_rounds = 365
    future_visits_delay = 30

    for envN, seedV in enumerate([3077083550, 538541279]):
        env = Environment(random_seed=seedV) if seedV is not None else Environment()
        print(f'Running with seed {env.get_seed()}')

        opt_price, opt_bid, profit = step1(env, prices, bids)

        bandit_env = BidBanditEnvironment(env, opt_price, bids, future_visits_delay)

        expected_profits = np.array([expected_profit(env, opt_price, b) for b in bids])
        gaps = np.max(expected_profits) - expected_profits
        rescaled_gaps = gaps/np.max(gaps)
        clairvoyant_cumulative_profits = np.cumsum([np.max(expected_profits)]*n_rounds)

        ucb_leaner = UCBOptimalBidLearner(bandit_env, lambda r: r % 400 == 401)
        ucb_leaner.learn(n_rounds)
        ucb_cumulative_profits = ucb_leaner.compute_cumulative_exp_profits(expected_profits)

        env.print_summary()

        for name, learner in [("UCB", ucb_leaner)]:
            print(name)
            print("Bid:               " + " ".join(f'{p:10d}' for p in bandit_env.bids))
            print("Projected profits: " + " ".join(f'{p:10.2f}' for p in learner.compute_projected_profits()))
            print("Expected profits:  " + " ".join(f'{p:10.2f}' for p in learner.compute_expected_profits()))
            print("Averages:          " + " ".join(f'{p:10.2f}' for p in learner.compute_average_new_clicks_per_arm()))
            print("Number of pulls:   " + " ".join(f'{p:10d}' for p in learner.get_number_of_pulls()))
            print()

        print(f"Optimal bid is {opt_bid} with expected profit {profit:.2f}")

        print("Bid:      " + " ".join(f'{p:10d} ' for p in bandit_env.bids))
        print("Expected: " + " ".join(f'{p:10.2f} ' for p in expected_profits))
        print("Gaps:     " + " ".join(f'{p:10.2f} ' for p in gaps))
        print("Rescaled: " + " ".join(f'{p*100:10.2f}%' for p in rescaled_gaps))

        if interactive:
            plot_results(["ucb"], [ucb_cumulative_profits], clairvoyant_cumulative_profits, n_rounds, smooth=False)

        table_data = [['Bid']]
        for p in bids:
            table_data.append([f'{p:.2f}'])

        table_data[0].append('True expected')
        for i, p in enumerate(bids):
            table_data[i + 1].append(f'{expected_profits[i]:.2f}')

        table_data[0].append('Gaps')
        for i, p in enumerate(bids):
            table_data[i + 1].append(f'{gaps[i]:.2f}')

        table_data[0] += ['Pulls', 'Learner expected']
        for i, (n, e) in enumerate(zip(ucb_leaner.get_number_of_pulls(), ucb_leaner.compute_expected_profits())):
            table_data[i + 1].append(f'{n}')
            table_data[i + 1].append(f'{e:.2f}')

        table = AsciiTable(table_data)
        for i in range(len(table_data[0])):
            table.justify_columns[i] = 'right'
        print(table.table)

        dir = '../../../report/figures/step5'

        with open(dir+f'/output{envN}.txt', 'w', encoding='utf8') as output_file:
            output_file.write(f' Seed: {env.get_seed()}\n')
            output_file.write(table.table)
            output_file.write(f'\n\n Optimal price: {opt_price:.2f}, Optimal bid: {opt_bid:.2f}')

        plot_results(["UCB"],
                     [ucb_cumulative_profits],
                     clairvoyant_cumulative_profits, n_rounds,
                     dest_file_path=dir+f'/plot{envN}')

        plot_everything(environment=env, path=dir+f'/env{envN}')


if __name__ == "__main__":
    main()
