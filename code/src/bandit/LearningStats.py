from code.src.bandit.BanditEnvironment import BanditEnvironment
import matplotlib.pyplot as plt
import numpy as np


def plot_results(names, rewards, regrets, clairvoyant, n_rounds):
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].set_title("Cumulative rewards")
    ax[1].set_title("Regret")

    x = [i for i in range(n_rounds + 1)]
    ax[0].plot(x, clairvoyant, 'green')

    for i in range(len(names)):
        alg_y = rewards[i]
        regret = regrets[i]

        ax[0].plot(x, alg_y)
        ax[1].plot(x, regret)

    ax[1].legend(names)
    names.insert(0, "clairvoyant")
    ax[0].legend(names)
    plt.show()
