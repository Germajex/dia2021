import matplotlib.pyplot as plt
import numpy as np

from src.Environment import Environment


# path must include the NAME and the EXTENSION of the image
def plot_new_clicks(environment: Environment, path=None):
    bids = np.linspace(1, 100, 101)
    plt.title(f"NEW CLICKS\nseed: {environment.get_seed()}")
    plt.xlabel("Bid")

    for c in environment.get_classes():
        new_clicks = [environment.get_dist_new_clicks().mean(c, bid)
                      for bid in bids]
        plt.plot(bids, new_clicks, label=c.get_name())

    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()


# path must include the NAME and the EXTENSION of the image
def plot_future_visits(environment: Environment, path=None):
    x_future_visits = np.linspace(1, 100, 1000)
    plt.title(f"FUTURE VISITS\nseed: {environment.get_seed()}")

    for c in environment.get_classes():
        future_visits = [environment.get_dist_future_visits().mean(c)] * len(x_future_visits)  # mean è uno scalare
        plt.plot(x_future_visits, future_visits, label=c.get_name())

    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()


# path must include the NAME and the EXTENSION of the image
def plot_clicks_converted(environment: Environment, path=None):
    prices = np.linspace(1, 100, 1000)
    plt.title(f"CLICKS CONVERTED\nseed: {environment.get_seed()}")
    plt.xlabel("Price")

    for c in environment.get_classes():
        clicks_converted = [environment.get_dist_click_converted().mean(c, price)
                            for price in prices]
        plt.plot(prices, clicks_converted, label=c.get_name())

    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()


# plots each chart for every class (path must include the NAME and the EXTENSION of the image)
def plot_everything(environment: Environment, path=None):
    bids = np.linspace(1, 100, 101)
    x_future_visits = np.linspace(1, 100, 1000)
    prices = np.linspace(1, 100, 1000)
    theta1 = environment.feature_1_likelihood
    theta2 = environment.feature_2_likelihood

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout(pad=5.0)
    fig.suptitle(f"Seed: {environment.get_seed()}")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fontsize = 12

    for color, c in zip(colors, environment.get_classes()):
        new_clicks = [environment.get_dist_new_clicks().mean(c, bid)
                      for bid in bids]
        future_visits = [environment.get_dist_future_visits().mean(c)] * len(x_future_visits)  # mean è uno scalare

        clicks_converted = [environment.get_dist_click_converted().mean(c, price)
                            for price in prices]

        axs[0][0].plot(bids, new_clicks, label=c.get_name(), color=color)
        axs[0][0].legend()
        axs[0][1].plot(x_future_visits, future_visits, label=c.get_name(), color=color)
        axs[0][1].legend()
        axs[1][0].plot(prices, clicks_converted, label=c.get_name(), color=color)
        axs[1][0].legend()

        for comb in c.features:
            f1, f2 = comb
            f1_l, f1_h = (1 - theta1, 1) if f1 else (0, 1 - theta1)
            f2_l, f2_h = (1 - theta2, 1) if f2 else (0, 1 - theta2)
            x = [f1_l, f1_l, f1_h, f1_h]
            y = [f2_l, f2_h, f2_h, f2_l]
            cx = (f1_l + f1_h) / 2
            cy = (f2_l + f2_h) / 2
            axs[1][1].fill(x, y, color=color)
            axs[1][1].text(cx, cy, c.name, color='k', fontsize=fontsize, ha='center', va='center')
            axs[1][1].text(cx, cy-0.1, f'l={environment.get_features_comb_likelihood(comb):.2f}', color='k', fontsize=fontsize, ha='center', va='center')

    axs[1][1].vlines(x=0, color='k', ymin=0.0, ymax=1.0)
    axs[1][1].hlines(y=0, color='k', xmin=0, xmax=1)
    axs[1][1].vlines(x=1, color='k', ymin=0.0, ymax=1.0)
    axs[1][1].hlines(y=1, color='k', xmin=0, xmax=1)
    axs[1][1].vlines(x=1 - theta1, color='k', ymin=0.0, ymax=1.0)
    axs[1][1].hlines(y=1 - theta2, color='k', xmin=0, xmax=1)
    axs[1][1].set_xlabel("Feature 1", fontsize=fontsize)
    axs[1][1].set_ylabel("Feature 2", fontsize=fontsize)

    axs[1][1].set_xticks([(1 - theta1) / 2, 1 - theta1 / 2])
    axs[1][1].set_xticklabels(['False', 'True'])
    axs[1][1].set_yticks([(1 - theta2) / 2, 1 - theta2 / 2])
    axs[1][1].set_yticklabels(['False', 'True'])

    axs[0][0].set_xlabel("Bid", fontsize=fontsize)
    axs[0][0].set_ylabel("E[ New clicks ]", fontsize=fontsize)

    axs[0][1].set_xlabel("")
    axs[0][1].set_ylabel("E[ Future visits ]", fontsize=fontsize)

    axs[1][0].set_xlabel("Price", fontsize=fontsize)
    axs[1][0].set_ylabel("E[ Click converted ]", fontsize=fontsize)

    if path is not None:
        plt.savefig(path)

    plt.show()


if __name__ == "__main__":
    test_environment = Environment(3626227578)

    # plot_new_clicks(test_environment)
    # plot_future_visits(test_environment)
    # plot_clicks_converted(test_environment)
    test_environment.print_summary()
    plot_everything(test_environment)
