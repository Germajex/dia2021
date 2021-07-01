import numpy as np

from src.Environment import Environment
from src.algorithms import optimal_bid_for_price, expected_profit


class BidBanditEnvironment:
    def __init__(self, environment: Environment, price, bids, future_visits_delay: int):
        # Init local vars
        self.rng = environment.rng

        self.n_arms = len(bids)
        self.price = price
        self.bids = bids
        self.env = environment
        self.future_visits_delay = future_visits_delay

        self.future_visits_queue = None
        self.reset_state()

        self.current_round = 0

    def reset_state(self):
        self.future_visits_queue = [(None, {comb: 0 for comb in
                                            self.env.get_features_combinations()})] * self.future_visits_delay

    def pull_arm_not_discriminating(self, arm: int):
        arm_strategy = {comb: arm for comb in self.env.get_features_combinations()}

        new_clicks, purchases, tot_cost_per_clicks, \
            (past_arm_strategy, past_future_visits) = self._inner_pull_arm(arm_strategy)

        past_pulled_arm = None if past_arm_strategy is None else past_arm_strategy[(False, False)]

        return sum(new_clicks.values()), sum(purchases.values()), sum(tot_cost_per_clicks.values()), \
            (past_pulled_arm, sum(past_future_visits.values()))

    def _inner_pull_arm(self, arm_strategy):
        bidding_strategy = {c: self.bids[a] for c, a in arm_strategy.items()}

        auctions, new_clicks, purchases, tot_cost_per_clicks, \
            new_future_visits = self.env.simulate_one_day_fixed_price(self.price, bidding_strategy)

        self.future_visits_queue.append((arm_strategy, new_future_visits))
        past_future_visits = self.future_visits_queue.pop(0)

        self.current_round += 1

        return new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def margin(self):
        return self.env.margin(self.price)

    def get_features_combinations(self):
        return self.env.get_features_combinations()

    def get_clairvoyant_best_bid_not_discriminating(self):
        optimal_bid = optimal_bid_for_price(self.env, self.bids, self.price)
        return optimal_bid

    def get_clairvoyant_optimal_expected_profit_not_discriminating(self):
        optimal_bid = self.get_clairvoyant_best_bid_not_discriminating()
        round_profit = expected_profit(self.env, self.price, optimal_bid)
        return round_profit

    def get_clairvoyant_cumulative_profits_not_discriminating(self, n_rounds):
        round_profit = self.get_clairvoyant_optimal_expected_profit_not_discriminating()
        return np.cumsum([round_profit] * n_rounds)
