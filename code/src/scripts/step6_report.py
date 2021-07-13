import numpy as np
from terminaltables import AsciiTable

from src.Environment import Environment
from src.algorithms import step1, expected_profit
from src.bandit.banditEnvironments.BidBanditEnvironment import BidBanditEnvironment
from src.bandit.LearningStats import plot_results
from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.learner.ucb.UCBOptimalBidLearner import UCBOptimalBidLearner
from src.bandit.learner.ucb.UCBOptimalJointLearner import UCBOptimalJointLearner
from src.constants import _Const
from src.scripts.environment_plotter import plot_everything


def main():
    CONST = _Const()

    interactive = False
    prices = np.linspace(10, 100, num=10, dtype=np.int64)
    bids = np.linspace(1, CONST.BID_MAX+10, num=10, dtype=np.int32)

    n_rounds = 365
    future_visits_delay = 30

    for envN, seedV in enumerate([None, None, None]):
        env = Environment(random_seed=seedV) if seedV is not None else Environment()
        print(f'Running with seed {env.get_seed()}')

        opt_price, opt_bid, profit = step1(env, prices, bids)

        bandit_env = JointBanditEnvironment(env, prices, bids, future_visits_delay)

        expected_profits = np.array([[expected_profit(env, p, b)
                                     for b in bids]
                                     for p in prices])

        gaps = np.max(expected_profits) - expected_profits
        rescaled_gaps = gaps/np.max(gaps)
        clairvoyant_cumulative_profits = np.cumsum([np.max(expected_profits)]*n_rounds)

        ucb_learner = UCBOptimalJointLearner(bandit_env, lambda r: r % 400 == 401)
        ucb_learner.learn(n_rounds)
        ucb_cumulative_profits = ucb_learner.compute_cumulative_exp_profits(expected_profits)

        env.print_summary()

        print(f'Optimal price is {opt_price} and optimal bid is {opt_bid}')

        ucb_recap = ucb_learner.get_pulled_arms_recap()
        print(f'Bids ->  | ' + ' '.join(f'{bandit_env.bids[b]:5d}' for b in range(ucb_learner.n_arms_bid)))
        print(f'Prices v | ' + '-' * 6 * ucb_learner.n_arms_bid)
        for arm_p in range(ucb_learner.n_arms_price):
            print(f'{bandit_env.prices[arm_p]:3d}      | ' + ' '.join(f'{p:5d}' for p in ucb_recap[arm_p]))

        if interactive:
            plot_results(["UCB"], [ucb_cumulative_profits], clairvoyant_cumulative_profits, n_rounds)

        table_pull_data = [['Price\Bid'] + [f'{b:4.2f}' for b in bids]]
        for p in prices:
            table_pull_data.append([f'{p:.2f}'])

        for p_i, p in enumerate(prices):
            for b_i, b in enumerate(bids):
                table_pull_data[p_i+1].append(ucb_recap[p_i][b_i])

        table_pull = AsciiTable(table_pull_data)
        for i in range(len(table_pull_data[0])):
            table_pull.justify_columns[i] = 'right'
        print(table_pull.table)

        table_gap_data = [['Price\Bid'] + [f'{b:4.2f}' for b in bids]]
        for p in prices:
            table_gap_data.append([f'{p:.2f}'])

        for p_i, p in enumerate(prices):
            for b_i, b in enumerate(bids):
                table_gap_data[p_i+1].append(f'{gaps[p_i][b_i]:.0f}')

        table_gap = AsciiTable(table_gap_data)
        for i in range(len(table_gap_data[0])):
            table_gap.justify_columns[i] = 'right'
        print(table_gap.table)

        dir = '../../../report/figures/step6'

        with open(dir+f'/output{envN}.txt', 'w', encoding='utf8') as output_file:
            output_file.write(f' Seed: {env.get_seed()}\n')
            output_file.write(f' UCB number of pulls:\n')
            output_file.write(table_pull.table)
            output_file.write(f'\n\n Optimal price: {opt_price:.2f}, Optimal bid: {opt_bid:.2f}\n\n')
            output_file.write('Gaps from the optimal arm:\n')
            output_file.write(table_gap.table)

        plot_results(["UCB"],
                     [ucb_cumulative_profits],
                     clairvoyant_cumulative_profits, n_rounds,
                     dest_file_path=dir+f'/plot{envN}')

        plot_everything(environment=env, path=dir+f'/env{envN}')


if __name__ == "__main__":
    main()
