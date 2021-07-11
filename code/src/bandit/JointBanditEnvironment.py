import numpy as np

from src.Environment import Environment
from src.algorithms import step1, expected_profit


class JointBanditEnvironment:
    def __init__(self, environment: Environment, prices, bids, future_visits_delay: int):
        self.rng = environment.rng

        self.n_arms_price = len(prices)
        self.n_arms_bid = len(bids)
        self.prices = prices
        self.bids = bids
        self.env = environment
        self.future_visits_delay = future_visits_delay

        self.future_visits_queue = None
        self.reset_state()

        self.current_round = 0

    def reset_state(self):
        self.future_visits_queue = [((None, None), {comb: 0 for comb in
                                            self.env.get_features_combinations()})] * self.future_visits_delay
        self.current_round = 0

    def pull_arm_not_discriminating(self, price_arm: int, bid_arm: int):
        price_arm_strategy = {comb: price_arm for comb in self.env.get_features_combinations()}
        bid_arm_strategy = {comb: bid_arm for comb in self.env.get_features_combinations()}

        auctions, new_clicks, purchases, tot_cost_per_clicks, \
            ((past_price_arm_strategy, past_bid_arm_strategy), past_future_visits) = self._inner_pull_arm(price_arm_strategy, bid_arm_strategy)

        past_price_pulled_arm = None if past_price_arm_strategy is None else past_price_arm_strategy[(False, False)]
        past_bid_pulled_arm = None if past_bid_arm_strategy is None else past_bid_arm_strategy[(False, False)]

        return sum(auctions.values()), sum(new_clicks.values()), sum(purchases.values()), sum(tot_cost_per_clicks.values()), \
            ((past_price_pulled_arm, past_bid_pulled_arm), sum(past_future_visits.values()))

    def _inner_pull_arm(self, price_arm_strategy, bid_arm_strategy):
        bidding_strategy = {c: self.bids[a] for c, a in bid_arm_strategy.items()}
        pricing_strategy = {c: self.prices[a] for c, a in price_arm_strategy.items()}

        auctions, new_clicks, purchases, tot_cost_per_clicks, \
            new_future_visits, _ = self.env.simulate_one_day(pricing_strategy, bidding_strategy)

        self.future_visits_queue.append(((price_arm_strategy, bid_arm_strategy), new_future_visits))
        past_future_visits = self.future_visits_queue.pop(0)

        self.current_round += 1

        return auctions, new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def margin(self, arm_price):
        return self.env.margin(self.prices[arm_price])

    def get_features_combinations(self):
        return self.env.get_features_combinations()

    def get_clairvoyant_optimal_expected_profit_not_discriminating(self):
        _, _, profit = step1(self.env, self.prices, self.bids)
        return profit

    def get_clairvoyant_cumulative_profits_not_discriminating(self, n_rounds):
        round_profit = self.get_clairvoyant_optimal_expected_profit_not_discriminating()
        return np.cumsum([round_profit] * n_rounds)