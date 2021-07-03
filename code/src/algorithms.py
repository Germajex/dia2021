import numpy as np


def simple_class_profit(margin, new_clicks, conversion_rate, future_visits, cost_per_click):
    return new_clicks * (margin * conversion_rate * (1 + future_visits) - cost_per_click)


def expected_profit(env, p, b, classes=None):
    m = env.margin
    n = env.distNewClicks.mean
    r = env.distClickConverted.mean
    f = env.distFutureVisits.mean
    k = env.distCostPerClick.mean

    C = classes if classes is not None else env.classes

    profit = sum(
        simple_class_profit(m(p), n(c,b), r(c, p), f(c), k(c, b))
        for c in C
    )

    return profit


def optimal_price_for_bid(env, prices, bid, classes=None):
    opt_p_index = np.argmax([
        expected_profit(env, p, bid, classes)
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
    median_b = bids[len(bids) // 2]

    optimal_price = optimal_price_for_bid(env, prices, median_b)
    optimal_bid = optimal_bid_for_price(env, bids, optimal_price)

    profit = expected_profit(env, optimal_price, optimal_bid)

    return optimal_price, optimal_bid, profit
