import numpy as np

from src.Environment import Environment
from src.algorithms import optimal_price_for_bid, simple_class_profit
from src.CustomerClass import CustomerClass


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class BanditEnvironment:
    def __init__(self, environment: Environment, prices, bid, future_visits_delay: int):
        # Init local vars
        self.n_arms = len(prices)
        self.prices = prices
        self.bid = bid
        self.env = environment
        self.future_visits_delay = future_visits_delay

        self.future_visits_queue = None
        self.reset_state()

        self.current_round = 0

    def pull_arm_not_discriminating(self, arm: int):
        new_clicks, purchases, tot_cost_per_clicks, past_future_visits = self._inner_pull_arm(arm)

        return sum(new_clicks), sum(purchases), sum(tot_cost_per_clicks), \
            (past_future_visits[0], sum(past_future_visits[1]))

    def pull_arm_discriminating(self, arm: int):
        print('')

    def _inner_pull_arm(self, arm: int):
        price = self.prices[arm]

        new_clicks = [self.env.distNewClicks.sample(customer_class=c, bid=self.bid)
                      for c in self.env.classes]

        purchases = [self.env.distClickConverted.sample_n(c, price, new_clicks[i])
                     for i, c in enumerate(self.env.classes)]

        tot_future_visits = tuple(
            [sum(self.env.distFutureVisits.sample_n(c, purchases[i]))
             for i, c in enumerate(self.env.classes)]
        )
        self.future_visits_queue.append((arm,  tot_future_visits))

        tot_cost_per_clicks = [
            sum(self.env.distCostPerClick.sample_n(c, self.bid, new_clicks[i]))
            for i, c in enumerate(self.env.classes)
        ]

        past_future_visits = self.future_visits_queue.pop(0)

        self.current_round += 1

        return new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def reset_state(self):
        self.future_visits_queue = [(None, (0, 0, 0))] * self.future_visits_delay

    def margin(self, arm:int):
        return self.env.margin(self.prices[arm])
