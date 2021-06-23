import matplotlib.pyplot as plt
import numpy as np

from src.Environment import Environment


def plot_new_clicks(environment: Environment):
    bids = np.linspace(1, 100, 101)
    plt.title(f"NEW CLICKS\nseed: {environment.get_seed()}")
    plt.xlabel("Bid")

    for c in environment.get_classes():
        new_clicks = [environment.get_dist_new_clicks().mean(c, bid)
                      for bid in bids]
        plt.plot(bids, new_clicks, label=c.get_name())

    plt.legend()
    plt.show()


def plot_future_visits(environment: Environment):
    x_future_visits = np.linspace(1, 100, 1000)
    plt.title(f"FUTURE VISITS\nseed: {environment.get_seed()}")

    for c in environment.get_classes():
        future_visits = [environment.get_dist_future_visits().mean(c)] * len(x_future_visits)  # mean è uno scalare
        plt.plot(x_future_visits, future_visits, label=c.get_name())

    plt.legend()
    plt.show()


def plot_clicks_converted(environment: Environment):
    prices = np.linspace(1, 100, 1000)
    plt.title(f"CLICKS CONVERTED\nseed: {environment.get_seed()}")
    plt.xlabel("Price")

    for c in environment.get_classes():
        clicks_converted = [environment.get_dist_click_converted().mean(c, price)
                            for price in prices]
        plt.plot(prices, clicks_converted, label=c.get_name())

    plt.legend()
    plt.show()


# plots each chart for every class
def plot_everything(environment: Environment):
    bids = np.linspace(1, 100, 101)
    x_future_visits = np.linspace(1, 100, 1000)
    prices = np.linspace(1, 100, 1000)

    fig, axs = plt.subplots(3)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout(pad=5.0)
    fig.suptitle(f"Seed: {environment.get_seed()}")

    for c in environment.get_classes():
        new_clicks = [environment.get_dist_new_clicks().mean(c, bid)
                      for bid in bids]
        future_visits = [environment.get_dist_future_visits().mean(c)] * len(x_future_visits)  # mean è uno scalare

        clicks_converted = [environment.get_dist_click_converted().mean(c, price)
                            for price in prices]

        axs[0].plot(bids, new_clicks, label=c.get_name())
        axs[0].legend()
        axs[1].plot(x_future_visits, future_visits, label=c.get_name())
        axs[1].legend()
        axs[2].plot(prices, clicks_converted, label=c.get_name())
        axs[2].legend()

    x_labels = ["Bid", "", "Price"]
    y_labels = ["NEW CLICKS", "FUTURE VISITS", "CLICKS CONVERTED"]
    i = 0
    for ax in axs.flat:
        ax.set_xlabel(f'{x_labels[i]}', fontsize=12)
        ax.set_ylabel(f'{y_labels[i]}', fontsize=12)
        i += 1

    plt.show()


test_environment = Environment()
# plot_new_clicks(test_environment)
# plot_future_visits(test_environment)
# plot_clicks_converted(test_environment)
plot_everything(test_environment)
