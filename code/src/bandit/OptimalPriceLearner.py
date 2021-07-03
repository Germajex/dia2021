from typing import List

import numpy as np
from src.algorithms import simple_class_profit
from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment


class OptimalPriceLearner:
    def __init__(self, env: PriceBanditEnvironment):
        self.env = env
        self.n_arms = self.env.n_arms
        self.future_visits_per_arm = [[] for i in range(self.n_arms)]
        self.purchases_per_arm = [[] for i in range(self.n_arms)]
        self.new_clicks_per_arm = [[] for i in range(self.n_arms)]
        self.tot_cost_per_click = 0
        self.current_round = 0
        self.expected_profits = []

        self.pulled_arms = []

    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        arm = self.choose_next_arm()
        self.pull_from_env(arm=arm)

    def choose_next_arm(self):
        return int(np.argmax(self.compute_projected_profits()))

    def compute_projected_profits(self):
        new_clicks = self.compute_new_clicks()
        margin = np.array([self.env.margin(a) for a in range(self.n_arms)])
        crs = self.compute_projection_conversion_rates()
        future_visits = self.compute_future_visits_per_arm()
        cost_per_click = self.tot_cost_per_click / self.sum_ragged_matrix(self.new_clicks_per_arm)

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    def compute_expected_profits(self):
        new_clicks = self.compute_new_clicks()
        margin = np.array([self.env.margin(a) for a in range(self.n_arms)])
        crs = self.get_average_conversion_rates()
        future_visits = self.compute_future_visits_per_arm()
        cost_per_click = self.tot_cost_per_click / self.sum_ragged_matrix(self.new_clicks_per_arm)

        expected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return expected_profit

    def compute_expected_profit_one_round(self, new_clicks, purchases, tot_cost_per_clicks, arm):
        cr = purchases / new_clicks
        future = self.compute_future_visits()
        margin = self.env.margin(arm)

        expected_profit = new_clicks * (margin * cr * (1 + future)) - tot_cost_per_clicks
        return expected_profit

    def compute_cumulative_profits(self):
        return np.cumsum(self.expected_profits)

    def compute_future_visits(self):
        successes_sum = 0

        for arm in range(self.n_arms):
            complete_samples = len(self.future_visits_per_arm[arm])
            successes_sum += np.sum(self.purchases_per_arm[arm][:complete_samples])

        return self.sum_ragged_matrix(self.future_visits_per_arm) / successes_sum

    def compute_future_visits_per_arm(self):
        res = []
        for arm in range(self.n_arms):
            complete_samples = len(self.future_visits_per_arm[arm])
            future_visits = np.sum(self.future_visits_per_arm[arm])
            purchases = np.sum(self.purchases_per_arm[arm][:complete_samples])
            future_visits_per_purchase = future_visits / purchases
            res.append(future_visits_per_purchase)

        return np.array(res)

    def compute_new_clicks(self):
        return self.average_ragged_matrix(self.new_clicks_per_arm)

    def compute_projection_conversion_rates(self):
        raise NotImplementedError

    def get_average_conversion_rates(self):
        raise NotImplementedError

    def get_number_of_pulls(self):
        return [len(a) for a in self.new_clicks_per_arm]

    def round_robin(self):
        while not all(self.future_visits_per_arm):
            arm = self.current_round % self.n_arms
            self.pull_from_env(arm)

    def pull_from_env(self, arm: int):
        new_clicks, purchases, tot_cost_per_clicks, \
            (old_a, visits) = self.env.pull_arm_not_discriminating(arm)

        self.new_clicks_per_arm[arm].append(new_clicks)
        self.purchases_per_arm[arm].append(purchases)
        self.tot_cost_per_click += tot_cost_per_clicks

        if old_a is not None:
            self.future_visits_per_arm[old_a].append(visits)
            self.expected_profits.append(self.compute_expected_profit_one_round(new_clicks, purchases,
                                                                                tot_cost_per_clicks, arm))

        self.current_round += 1
        self.pulled_arms.append(arm)

        return new_clicks, purchases, tot_cost_per_clicks, (old_a, visits)

    @staticmethod
    def average_ragged_matrix(mat):
        return OptimalPriceLearner.sum_ragged_matrix(mat) / OptimalPriceLearner.count_ragged_matrix(mat)

    @staticmethod
    def count_ragged_matrix(mat):
        return np.sum(np.fromiter((len(r) for r in mat), dtype=np.int32))

    @staticmethod
    def sum_ragged_matrix(mat: List[List[int]]):
        return np.sum(np.fromiter((np.sum(r) for r in mat), dtype=np.float64))

    def compute_cumulative_regr(self, suboptimality_gaps):
        return np.cumsum([suboptimality_gaps[a] for a in self.pulled_arms])

    def compute_cumulative_exp_profits(self, expected_profits):
        return np.cumsum([expected_profits[a] for a in self.pulled_arms])
