import numpy as np

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
