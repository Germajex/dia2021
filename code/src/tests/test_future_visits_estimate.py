import numpy as np

from src.Environment import Environment


def main():
    samples_per_pair = 1000

    prices = np.arange(10, 101, 10)
    bids = np.arange(10, 101, 10)
    env = Environment()

    future_visits = np.zeros((len(prices), len(bids)))
    purchases = np.zeros((len(prices), len(bids)))
    for j in range(samples_per_pair):
        if not j % 10:
            print(f'Simulating: {j/samples_per_pair*100:6.2f}%')
        for p_i, price in enumerate(prices):
            for b_i, bid in enumerate(bids):
                _, _, purch, _, fut, _ = env.simulate_one_day_fixed_both(price, bid)
                purchases[p_i][b_i] += sum(purch.values())
                future_visits[p_i][b_i] += sum(fut.values())

    print(f'Simulating: 100.00%')

    print('    | ' + "  ".join(f'{bid:5d}' for bid in bids) + " |  aggr", flush=True)
    for p_i, price in enumerate(prices):
        print(f'{price:3d} | ' +
              "  ".join(f'{future_visits[p_i][b_i] / purchases[p_i][b_i]:5.2f}'
                        for b_i, bid in enumerate(bids)
                        ) + f" | {np.sum(future_visits[p_i]) / np.sum(purchases[p_i]):5.2f}"
              , flush=True)


if __name__ == '__main__':
    main()
