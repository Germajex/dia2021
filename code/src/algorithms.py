import numpy as np


def expected_profit(env, p, b):
    m = env.margin
    n = env.distNewClicks.mean
    r = env.distClickConverted.mean
    f = env.distFutureVisits.mean
    k = env.distCostPerClick.mean

    profit = m(p) * sum(
        n(c, b) * r(c, p) * (1 + f(c))
        for c in env.classes
    ) - k(b) * sum(
        n(c, b)
        for c in env.classes
    )

    return profit


def optimal_price_for_bid(env, prices, bid):
    opt_p_index = np.argmax([
        expected_profit(env, p, bid)
        for p in prices
    ])

    return prices[opt_p_index]


def optimal_bid_for_price(env, bids, price):
    opt_b_index = np.argmax([
        expected_profit(env, price, b)
        for b in bids
    ])
    return bids[opt_b_index]


def step1(env, prices, bids):
    median_b = bids[len(bids)//2]

    optimal_price = optimal_price_for_bid(env, prices, median_b)
    optimal_bid = optimal_bid_for_price(env, bids, optimal_price)

    profit = expected_profit(env, optimal_price, optimal_bid)

    return optimal_price, optimal_bid, profit
