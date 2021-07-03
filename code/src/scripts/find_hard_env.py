import numpy as np

from src.Environment import Environment
from src.algorithms import step1, expected_profit


def main():
    max_iter = 10000
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    for it in range(1,max_iter+1):
        env = Environment()
        opt_price, opt_bid, profit = step1(env, prices, bids)

        expected_profits = np.array([expected_profit(env, p, opt_bid) for p in prices])

        optimality_gaps = np.max(expected_profits) - expected_profits
        rescaled_gaps = optimality_gaps / np.max(expected_profits)

        if np.sort(rescaled_gaps)[2] < 0.05:
            print(f'Found environment with seed {env.get_seed()} and small gaps')
            break

        print(f'Completed iteration {it}/{max_iter}')


if __name__ == '__main__':
    main()
