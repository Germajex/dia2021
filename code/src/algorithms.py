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
    return expected_profit(env, p, b, [comb])


# start expected profit
def expected_profit(env, p, b, combinations=None):
    m = env.margin
    n = env.distNewClicks.mean_per_comb(b)
    r = env.distClickConverted.mean
    f = env.distFutureVisits.mean
    k = env.distCostPerClick.mean

    combs = combinations if combinations is not None else env.combinations

    profit = sum(
        simple_class_profit(margin=m(p),
                            new_clicks=n[comb],
                            conversion_rate=r(env.class_of_comb[comb], p),
                            future_visits=f(env.class_of_comb[comb]),
                            cost_per_click=k(env.class_of_comb[comb], b))
        for comb in combs
    )

    return profit


# end expected profit


def optimal_pricing_strategy_for_bid(env, prices, bid):
    strategy = {}
    for c in env.classes:
        opt_p = optimal_price_for_bid(env, prices, bid, combinations=c.features)
        for comb in c.features:
            strategy[comb] = opt_p

    return strategy


# start step 1
def step1(env, prices, bids, combinations=None):
    median_b = bids[len(bids) // 2]

    optimal_price = optimal_price_for_bid(env, prices, median_b, combinations)
    optimal_bid = optimal_bid_for_price(env, bids, optimal_price, combinations)

    profit = expected_profit(env, optimal_price, optimal_bid, combinations)

    return optimal_price, optimal_bid, profit


# end step 1

# start step 1 support
def optimal_price_for_bid(env, prices, bid, combinations=None):
    opt_p_index = np.argmax([
        expected_profit(env, p, bid, combinations=combinations)
        for p in prices
    ])

    return prices[opt_p_index]


def optimal_bid_for_price(env, bids, price, combinations=None):
    opt_b_index = np.argmax([
        expected_profit(env, price, b, combinations=combinations)
        for b in bids
    ])
    return bids[opt_b_index]
# end step 1 support
