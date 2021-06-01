import numpy as np


def margin(p):
    return p


def expected_profit(env, p, b):
    m = margin
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


def step1(env, prices, bids):
    median_b = bids[len(bids)//2]

    opt_p_index = np.argmax([
        expected_profit(env, p, median_b)
        for p in prices
    ])

    opt_p = prices[opt_p_index]

    opt_b_index = np.argmax([
        expected_profit(env, opt_p, b)
        for b in bids
    ])
    opt_b = bids[opt_b_index]

    profit = expected_profit(env, opt_p, opt_b)

    return opt_p, opt_b, profit


def optimize_all_known(classes, prices, bids):
    # Sets of items
    C = classes
    P = prices
    B = bids

    # Since we have no guarantees that the function has a unique local optima, we have to scan the whole "line"
    j = len(B) // 2

    # Optimize price
    p_ndx = np.argmax([np.sum([c.getRevenue(p, B[j]) for c in C]) for p in P])

    # Optimize bid
    b_ndx = np.argmax([np.sum([c.getRevenue(P[p_ndx], b) for c in C]) for b in B])

    result = np.sum([c.getRevenue(P[p_ndx], B[b_ndx]) for c in C])

    return P[p_ndx], B[b_ndx], result
