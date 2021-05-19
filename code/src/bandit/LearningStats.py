from code.src.bandit.BanditEnvironment import BanditEnvironment
import matplotlib.pyplot as plt
import numpy as np


def plot_results(names, rewards, prices, bids, n_rounds, env: BanditEnvironment):

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].set_title("Cumulative rewards")
    ax[1].set_title("Regret")

    clairvoyant = [env.get_optimal_reward(0, bids[0]) * t for t in
                   range(n_rounds+1)]

    x = [i for i in range(n_rounds+1)]
    ax[0].plot(x, clairvoyant, 'green')

    for i in range(len(names)):
        alg_y = rewards[i]
        regret = np.array(clairvoyant) - np.array(alg_y)

        ax[0].plot(x, alg_y)
        ax[1].plot(x, regret)

    ax[1].legend(names)
    names.insert(0, "clairvoyant")
    ax[0].legend(names)
    plt.show()
