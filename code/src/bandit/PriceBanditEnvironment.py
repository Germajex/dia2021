import numpy as np

from src.Environment import Environment
from src.algorithms import optimal_price_for_bid, expected_profit


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class PriceBanditEnvironment:
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

    def _inner_pull_arm(self, arm_strategy):
        pricing_strategy = {c: self.prices[a] for c, a in arm_strategy.items()}

        auctions, new_clicks, purchases, tot_cost_per_clicks, \
            new_future_visits, profit = self.env.simulate_one_day_fixed_bid(pricing_strategy, self.bid)

        self.future_visits_queue.append((arm_strategy, new_future_visits))
        past_future_visits = self.future_visits_queue.pop(0)

        self.current_round += 1

        return new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def reset_state(self):
        self.future_visits_queue = [(None, {comb: 0 for comb in
                                            self.env.get_features_combinations()})] * self.future_visits_delay
        self.current_round = 0

    def margin(self, arm: int):
        return self.env.margin(self.prices[arm])

    def get_features_combinations(self):
        return self.env.get_features_combinations()

    def get_clairvoyant_best_price_not_discriminating(self):
        optimal_price = optimal_price_for_bid(self.env, self.prices, self.bid)
        return optimal_price

    def get_clairvoyant_optimal_expected_profit_not_discriminating(self):
        optimal_price = self.get_clairvoyant_best_price_not_discriminating()
        round_profit = expected_profit(self.env, optimal_price, self.bid)
        return round_profit

    def get_clairvoyant_cumulative_profits_not_discriminating(self, n_rounds):
        round_profit = self.get_clairvoyant_optimal_expected_profit_not_discriminating()
        return np.cumsum([round_profit] * n_rounds)

    def get_clairvoyant_best_prices_discriminating(self):
        optimal_prices = [optimal_price_for_bid(self.env, self.prices, self.bid, [c]) for c in self.env.classes]
        return optimal_prices

    def get_clairvoyant_optimal_expected_profit_discriminating(self):
        optimal_prices = self.get_clairvoyant_best_prices_discriminating()
        round_profit = np.sum([expected_profit(self.env, optimal_prices[ndx], self.bid, [c]) for ndx, c in enumerate(self.env.classes)])
        return round_profit

    def get_clairvoyant_cumulative_profits_discriminating(self, n_rounds):
        round_profit = self.get_clairvoyant_optimal_expected_profit_discriminating()
        return np.cumsum([round_profit] * n_rounds)
