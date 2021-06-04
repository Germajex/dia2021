import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def plot_results(names, rewards, clairvoyant, n_rounds, smooth=False):
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].set_title("Cumulative rewards")
    ax[1].set_title("Regret")

    x = [i for i in range(n_rounds + 1)]
    ax[0].plot(x, clairvoyant, 'red')

    for i in range(len(names)):
        alg_y = rewards[i]
        regret = np.array(clairvoyant) - np.array(rewards[i])

        if smooth:
            # Smooth
            x_pre_smooth = np.linspace(0, n_rounds, n_rounds//30)
            bspline = interpolate.make_interp_spline(x, regret)
            regret_pre_smooth = bspline(x_pre_smooth)

            x_smooth = np.linspace(0, n_rounds, n_rounds)
            bspline = interpolate.make_interp_spline(x_pre_smooth, regret_pre_smooth)
            regret_smooth = bspline(x_smooth)

            ax[1].plot(x_smooth, regret_smooth)
        else:
            ax[1].plot(x, regret)

        ax[0].plot(x, alg_y)

    ax[1].legend(names)
    names.insert(0, "clairvoyant")
    ax[0].legend(names)
    plt.show()
