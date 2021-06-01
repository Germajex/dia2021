import numpy as np

from src.Environment import Environment
from src.algorithms import step1


def main():
    env = Environment()

    prices = np.linspace(1, 100, 1000)
    bids = np.linspace(1, 100, 1000)

    print(f'Computing the optimal price and bid with |P| = {len(prices)}, |B| = {len(bids)}')
    opt_price, opt_bid, profit = step1(env, prices, bids)
    print('Optimal price and bid found.')
    print(f'p = {opt_price}, b = {opt_bid}, expected profit = {profit}')

    """
    print('Computing the optimal price and bid with old algorithm:')

    opt_price, opt_bid, profit = optimize_all_known(env.classes, prices, bids)
    print('Optimal price and bid found.')
    print(f'p = {opt_price}, b = {opt_bid}, expected profit = {profit}')
    """


if __name__ == "__main__":
    main()
