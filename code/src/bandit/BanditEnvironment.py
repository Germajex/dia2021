import numpy as np

from src.Environment import Environment
from src.CustomerClass import CustomerClass


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class BanditEnvironment:
    def __init__(self, environment: Environment, prices, bid, future_visits_delay: int):
        # Init local vars
        self.rng = environment.rng

        self.n_arms = len(prices)
        self.prices = prices
        self.bid = bid
        self.env = environment
        self.future_visits_delay = future_visits_delay

        self.future_visits_queue = None
        self.reset_state()

        self.current_round = 0

    def pull_arm_not_discriminating(self, arm: int):
        arm_strategy = {comb: arm for comb in self.env.get_features_combinations()}

        new_clicks, purchases, tot_cost_per_clicks, \
        (past_arm_strategy, past_future_visits) = self._inner_pull_arm(arm_strategy)

        past_pulled_arm = None if past_arm_strategy is None else past_arm_strategy[(False, False)]

        return sum(new_clicks.values()), sum(purchases.values()), sum(tot_cost_per_clicks.values()), \
               (past_pulled_arm, sum(past_future_visits.values()))

    def pull_arm_discriminating(self, arm_strategy):
        new_clicks, purchases, tot_cost_per_clicks, past_future_visits = self._inner_pull_arm(arm_strategy)

        return new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def get_users_count(self, new_clicks, c: CustomerClass):
        features_comb_likelihoods = [
            self.env.get_features_comb_likelihood(f)
            for f in c.features
        ]

        features_comb_dist = np.array(features_comb_likelihoods) / sum(features_comb_likelihoods)

        users = self.rng.multinomial(new_clicks, features_comb_dist)
        return users

    def _inner_pull_arm(self, arm_strategy):
        pricing_strategy = {c: self.prices[a] for c, a in arm_strategy.items()}

        auctions, new_clicks, purchases, tot_cost_per_clicks, \
            new_future_visits = self.env.simulate_one_day(pricing_strategy, self.bid)

        self.future_visits_queue.append((arm_strategy, new_future_visits))
        past_future_visits = self.future_visits_queue.pop(0)

        self.current_round += 1

        return new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def reset_state(self):
        self.future_visits_queue = [(None, {comb: 0 for comb in
                                            self.env.get_features_combinations()})] * self.future_visits_delay

    def margin(self, arm: int):
        return self.env.margin(self.prices[arm])

    def get_features_combinations(self):
        return self.env.get_features_combinations()
