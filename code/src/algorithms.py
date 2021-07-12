import numpy as np


# start simple class profit
def simple_class_profit(margin, new_clicks, conversion_rate,
                        future_visits, cost_per_click):
    return new_clicks * (margin * conversion_rate * (1 + future_visits)
                         - cost_per_click)


# end simple class profit

def expected_profit_of_pricing_strategy(env, pricing_strategy, b):
    return sum(expected_profit_for_comb(env, p, b, comb)
               for comb, p in pricing_strategy.items())


def expected_profit_for_comb(env, p, b, comb):
    c = env.class_of_comb[comb]
    margin = env.margin(p)
    new_click = env.distNewClicks.mean_per_comb(b)[comb]
    conversion_rate = env.distClickConverted.mean(c, p)
    future_visits = env.distFutureVisits.mean(c)
    cost_per_click = env.distCostPerClick.mean(c, b)

    profit = simple_class_profit(margin, new_click, conversion_rate, future_visits, cost_per_click)

    return profit


# start expected profit
def expected_profit(env, p, b, classes=None):
    m = env.margin
    n = env.distNewClicks.mean
    r = env.distClickConverted.mean
    f = env.distFutureVisits.mean
    k = env.distCostPerClick.mean

    C = classes if classes is not None else env.classes

    profit = sum(
        simple_class_profit(m(p), n(c, b), r(c, p), f(c), k(c, b))
        for c in C
    )

    return profit


# end expected profit


def optimal_pricing_strategy_for_bid(env, prices, bid):
    strategy = {}
    for c in env.classes:
        opt_p = optimal_price_for_bid(env, prices, bid, [c])
        for comb in c.features:
            strategy[comb] = opt_p

    return strategy


# start step 1
def step1(env, prices, bids):
    median_b = bids[len(bids) // 2]

    optimal_price = optimal_price_for_bid(env, prices, median_b)
    optimal_bid = optimal_bid_for_price(env, bids, optimal_price)

    profit = expected_profit(env, optimal_price, optimal_bid)

    return optimal_price, optimal_bid, profit


# end step 1

# start step 1 support
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
# end step 1 support
