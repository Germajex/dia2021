import numpy as np
import matplotlib.pyplot as plt
from src.Environment import Environment
from src.algorithms import step1


def main():
    iterations = 100
    samples = 1000

    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    max_error = 1000
    errors_count = [0]*(2*max_error+1)
    cum_error = 0
    for test_n in range(1, samples+1):
        print(f'Running test n°{test_n:2d}/{samples}')
        env = Environment()
        opt_price, opt_bid, exp_profit = step1(env, prices, bids)

        pricing_strategy = {c:opt_price for c in env.get_features_combinations()}

        cum_profit = 0
        for __ in range(iterations):
            _, _, _, _, _, profit = env.simulate_one_day_fixed_bid(pricing_strategy, opt_bid)
            cum_profit += profit

        average_profit = cum_profit / iterations

        error = average_profit-exp_profit
        int_error = max(-max_error, min(max_error, round(error)))

        errors_count[int_error+max_error]+=1

        cum_error += error
        print(f'Run test n°{test_n:2d}/{samples} with error {error:.2f}')
    average_error = cum_error/samples

    print(f'Final average error: {average_error:.2f}')
    plt.scatter(range(-max_error, -max_error+len(errors_count)), errors_count)
    plt.show()


if __name__ == "__main__":
    main()
