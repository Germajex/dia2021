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

        self.latest_split = 0
        self.expected_profits = []

        self.context_structure: List[Context] = [
            self.context_creator(
                features=env.get_features_combinations(),
                arm_margin_function=env.margin,
                n_arms=self.n_arms,
                rng=self.env.rng
            )
        ]

        self.current_round = 0

    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        strategy = self.choose_next_strategy()
        self.pull_from_env(strategy=strategy)

        if self.current_round - self.latest_split >= 0:
            self.update_contexts()
        # print(f'Learned round {self.current_round+1}')

    def update_contexts(self):
        n_features = len(self.context_structure[0].features[0])

        for context in self.context_structure:
            current_lower = self.compute_context_expected_profit_lower_bound(context)
            new_structures = []
            for i in range(n_features):
                features_where_i_is_true = [f for f in context.features if f[i]]
                features_where_i_is_false = [f for f in context.features if not f[i]]

                # a valid split generates two non-empty contexts
                if features_where_i_is_false and features_where_i_is_true:
                    context_true = self.context_creator(features=features_where_i_is_true,
                                                        arm_margin_function=self.env.margin,
                                                        n_arms=self.n_arms,
                                                        rng=self.env.rng)

                    context_false = self.context_creator(features=features_where_i_is_false,
                                                         arm_margin_function=self.env.margin,
                                                         n_arms=self.n_arms,
                                                         rng=self.env.rng)
                    true_lower = self.compute_context_expected_profit_lower_bound(context_true)
                    false_lower = self.compute_context_expected_profit_lower_bound(context_false)

                    incentive = true_lower + false_lower - current_lower
                    if incentive > 0:
                        new_structure = list(self.context_structure)
                        new_structure.remove(context)
                        new_structure.append(context_true)
                        new_structure.append(context_false)

                        print(
                            f'Found feasible split at round {self.current_round} on feature {i}, incentive = {incentive}')
                        new_structures.append((incentive, new_structure, i))

            if new_structures:
                incentive, new_structure, feature = max(new_structures, key=lambda x: x[0])
                self.context_structure = new_structure
                print(f'Split context at round {self.current_round} on feature {feature}')
                self.latest_split = self.current_round

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
            self.expected_profits.append(self.compute_expected_profit_last_round(strategy))

        self.current_round += 1

    def get_contexts(self):
        return self.context_structure

    def compute_context_projected_profit(self, context: Context):
        return context.compute_projected_profit(self.new_clicks_per_comb_per_arm,
                                                self.purchases_per_comb_per_arm,
                                                self.tot_cost_per_click_per_comb,
                                                self.future_visits_per_comb_per_arm,
                                                self.current_round)

    def compute_context_expected_profit(self, context: Context):
        return context.compute_expected_profits(self.new_clicks_per_comb_per_arm,
                                                self.purchases_per_comb_per_arm,
                                                self.tot_cost_per_click_per_comb,
                                                self.future_visits_per_comb_per_arm)

    def compute_context_expected_profit_lower_bound(self, context: Context):
        return context.compute_expected_profit_lower_bound(self.new_clicks_per_comb_per_arm,
                                                           self.purchases_per_comb_per_arm,
                                                           self.tot_cost_per_click_per_comb,
                                                           self.future_visits_per_comb_per_arm,
                                                           self.current_round)

    def compute_expected_profit_last_round(self, strategy):
        profit = np.sum(context.compute_expected_profit_last_round(self.new_clicks_per_comb_per_arm,
                                                                   self.purchases_per_comb_per_arm,
                                                                   self.tot_cost_per_click_per_comb,
                                                                   self.future_visits_per_comb_per_arm,
                                                                   strategy[context.features[0]])
                        for context in self.context_structure)

        return profit

    def compute_cumulative_profits(self):
        return np.cumsum(self.expected_profits)

    def get_average_conversion_rates(self, context: Context):
        return context.get_average_conversion_rates(self.new_clicks_per_comb_per_arm, self.purchases_per_comb_per_arm)

    def get_context_number_of_pulls(self, context: Context):
        return context.get_number_of_pulls(self.new_clicks_per_comb_per_arm)
