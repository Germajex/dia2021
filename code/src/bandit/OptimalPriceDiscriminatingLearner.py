from typing import List

import numpy as np

from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.Context import Context


class OptimalPriceDiscriminatingLearner:
    def __init__(self, env: BanditEnvironment, context_creator):
        self.env = env
        self.n_arms = self.env.n_arms
        self.context_creator = context_creator

        combs = self.env.get_features_combinations()
        self.future_visits_per_comb_per_arm = {c: [[] for i in range(self.n_arms)] for c in combs}
        self.purchases_per_comb_per_arm = {c: [[] for i in range(self.n_arms)] for c in combs}
        self.new_clicks_per_comb_per_arm = {c: [[] for i in range(self.n_arms)] for c in combs}
        self.tot_cost_per_click_per_comb = {c: 0 for c in combs}

        self.context_structure: List[Context] = [
            context_creator(features=env.get_features_combinations(),
                            arm_margin_function=env.margin,
                            n_arms=self.n_arms)
        ]

        self.current_round = 0

    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        strategy = self.choose_next_strategy()
        self.pull_from_env(strategy=strategy)
        # print(f'Learned round {self.current_round+1}')
        # split contexts?

    def choose_next_strategy(self):
        strategy = {}
        for context in self.context_structure:
            arm = context.choose_next_arm(self.new_clicks_per_comb_per_arm,
                                          self.purchases_per_comb_per_arm,
                                          self.tot_cost_per_click_per_comb,
                                          self.future_visits_per_comb_per_arm,
                                          self.current_round)
            for comb in context.features:
                strategy[comb] = arm

        return strategy

    def round_robin(self):
        while not all(self.future_visits_per_comb_per_arm[(False, False)]):
            arm = self.current_round % self.n_arms
            strategy = {comb: arm for comb in self.env.get_features_combinations()}
            self.pull_from_env(strategy)

    def pull_from_env(self, strategy):
        new_clicks, purchases, tot_cost_per_clicks, \
        (past_arm_strategy, past_future_visits) = self.env.pull_arm_discriminating(strategy)

        for comb in self.env.get_features_combinations():
            self.new_clicks_per_comb_per_arm[comb][strategy[comb]].append(new_clicks[comb])
            self.purchases_per_comb_per_arm[comb][strategy[comb]].append(purchases[comb])
            self.tot_cost_per_click_per_comb[comb] += tot_cost_per_clicks[comb]

        if past_arm_strategy is not None:
            for comb, chosen_arm in past_arm_strategy.items():
                self.future_visits_per_comb_per_arm[comb][chosen_arm].append(past_future_visits[comb])

        self.current_round += 1

    def get_contexts(self):
        return self.context_structure

    def compute_context_projected_profit(self, context: Context):
        return context.compute_projected_profits(self.new_clicks_per_comb_per_arm,
                                                 self.purchases_per_comb_per_arm,
                                                 self.tot_cost_per_click_per_comb,
                                                 self.future_visits_per_comb_per_arm,
                                                 self.current_round)

    def compute_context_expected_profit(self, context: Context):
        return context.compute_expected_profits(self.new_clicks_per_comb_per_arm,
                                                self.purchases_per_comb_per_arm,
                                                self.tot_cost_per_click_per_comb,
                                                self.future_visits_per_comb_per_arm)

    def get_average_conversion_rates(self, context: Context):
        return context.get_average_conversion_rates(self.new_clicks_per_comb_per_arm, self.purchases_per_comb_per_arm)
