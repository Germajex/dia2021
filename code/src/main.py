from CustomerClassCreator import CustomerClassCreator
from numpy.random import default_rng
import numpy as np
from algorithms import optimize_all_known

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def run_all_known(classes, verbose=1):
    if verbose > 0:
        for c in classes:
            c.printSummary()

    prices = np.sort(rng.choice(range(1, 100), 10, replace=False))
    bids = np.sort(np.around(rng.choice(np.linspace(1, 40, 50), 10, replace=False), 2))

    if verbose > 0:
        print("\nPossible prices: {}".format(prices))
        print("Possible bids: {}".format(bids))

    revenues = []
    for p in prices:
        for b in bids:
            revenues.append(np.sum([c.getRevenue(p, b) for c in classes]))

    max_revenue = np.amax(revenues)
    res = optimize_all_known(classes, prices, bids)
    found_revenue = res[2]

    if max_revenue > found_revenue:
        outcome = False
        if verbose > 0:
            print("\n!!! FAIL !!!")
            print(f"Best revenue={max_revenue}, found revenue={found_revenue} for price={res[0]} and bid={res[1]}")
    else:
        outcome=True
        if verbose > 0:
            print("\nSUCCESS")
            print(f"Result found. Price={res[0]}, bid={res[1]}, revenue={found_revenue}")

    return outcome


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rng = default_rng()

    res = True
    i=0

    while res and i < 5000:
        creator = CustomerClassCreator()
        classes = creator.getNewClasses(rng)

        res = run_all_known(classes, verbose=0)

        # If something goes wrong, let's see better what's happening
        if not res:
            res = run_all_known(classes, verbose=1)
        i +=1

    print(res)

    if True:
        # Data for a three-dimensional line
        x = np.linspace(1, 80, 100)
        y = np.linspace(1, 80, 100)
        z = [
                [
                    np.sum([c.getRevenue(p, b) for c in classes])
                    for p in x
                ]
            for b in y
        ]
        opt = [
            x[np.argmax([
                    np.sum([c.getRevenue(p, b) for c in classes])
                    for p in x
                ])]
            for b in y
        ]
        graph = 2

        fig = plt.figure()

        if graph == 2:
            ax = plt.axes(projection='3d')
            ax.contour3D(x, y, z, 100)
            ax.set_xlabel('prices')
            ax.set_ylabel('bids')
            ax.set_zlabel('revenues')
            ax.plot(opt, y, 0)
            plt.show()
        if graph == 3:
            ax = plt.axes()
            ax.plot(y, opt)
            ax.set_xlabel('bids')
            ax.set_ylabel('optimal_p')
            plt.show()


