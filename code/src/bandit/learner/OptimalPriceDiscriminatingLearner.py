from typing import List

import numpy as np

from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.context import Context


class OptimalPriceDiscriminatingLearner:
    def __init__(self, env: PriceBanditEnvironment, context_creator, round_robins_per_cycle=1):
        self.env = env
        self.n_arms = self.env.n_arms
        self.context_creator = context_creator
        self.round_robins_per_cycle = round_robins_per_cycle

        combs = self.env.get_features_combinations()
        self.future_visits_per_comb_per_arm = {c: [[] for i in range(self.n_arms)] for c in combs}
        self.purchases_per_comb_per_arm = {c: [[] for i in range(self.n_arms)] for c in combs}
        self.new_clicks_per_comb_per_arm = {c: [[] for i in range(self.n_arms)] for c in combs}
        self.tot_cost_per_click_per_comb = {c: 0 for c in combs}

        self.strategies = []

        self.context_structure: List[Context] = [
            self.context_creator(
                features=env.get_features_combinations(),
                arm_margin_function=env.margin,
                n_arms=self.n_arms,
                rng=self.env.rng
            )
        ]

        self.current_round = 0

        self.next_round_robin_arm = 0
        self.state_is_explorative_rounds = True
        self.remaining_round_robins = 0
        self.remaining_normal_rounds = 0
        self.performed_round_robins = 0

    # start learning loop
    def learn(self, n_rounds: int):
        self.initial_round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        strategy = self.choose_next_strategy()
        self.pull_from_env(strategy=strategy)

        self.update_contexts()

    # end learning loop

    # start update context
    def update_contexts(self):
        for context in self.context_structure:
            possible_splits = self.compute_convenient_splits(context)
            if possible_splits:
                incentive, new_structure, feature = min(possible_splits,
                                                        key=lambda x: x[0])
                self.context_structure = new_structure
                print(f'Split context at round {self.current_round} '
                      f'on feature {feature}')

    # end update context

    # start convenient splits
    def compute_convenient_splits(self, context):
        current_lower = self.compute_context_expected_profit_lower_bound(context)
        new_structures = []

        for feature_n, context_true, context_false in self.compute_possible_splits(
                context
        ):
            true_lower = self.compute_context_expected_profit_lower_bound(
                context_true)
            false_lower = self.compute_context_expected_profit_lower_bound(
                context_false)

            incentive = true_lower + false_lower - current_lower
            if incentive > 0:
                new_structure = list(self.context_structure)
                new_structure.remove(context)
                new_structure.append(context_true)
                new_structure.append(context_false)

                print(
                    f'Found convenient split at round {self.current_round} '
                    f'on feature {feature_n}, incentive = {incentive:.2f}')
                new_structures.append((incentive, new_structure, feature_n))
        return new_structures

    # end convenient splits

    # start possible splits
    def compute_possible_splits(self, context):
        n_features = len(self.context_structure[0].features[0])
        res = []
        for i in range(n_features):
            features_where_i_is_true = [f for f in context.features if f[i]]
            features_where_i_is_false = [f for f in context.features if not f[i]]

            # a valid split generates two non-empty context
            if features_where_i_is_false and features_where_i_is_true:
                context_true = self.context_creator(features=features_where_i_is_true,
                                                    arm_margin_function=self.env.margin,
                                                    n_arms=self.n_arms,
                                                    rng=self.env.rng)

                context_false = self.context_creator(features=features_where_i_is_false,
                                                     arm_margin_function=self.env.margin,
                                                     n_arms=self.n_arms,
                                                     rng=self.env.rng)
                res.append((i, context_true, context_false))

        return res

    # end possible splits

    # start choose next explorative
    def choose_next_strategy_explorative(self):
        strategy = {}
        for context in self.context_structure:
            could_be_split = len(context.features) > 1

            if could_be_split:
                arm = self.next_round_robin_arm
            else:
                arm = context.choose_next_arm(self.new_clicks_per_comb_per_arm,
                                              self.purchases_per_comb_per_arm,
                                              self.tot_cost_per_click_per_comb,
                                              self.future_visits_per_comb_per_arm,
                                              self.current_round)
            for comb in context.features:
                strategy[comb] = arm

        return strategy

    # end choose next explorative

    # start choose next strategy
    def choose_next_strategy(self):
        if self.state_is_explorative_rounds:
            strategy = self.choose_next_strategy_explorative()
        else:
            strategy = self.choose_next_strategy_normal()
        return strategy

    def choose_next_strategy_normal(self):
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

    # end choose next strategy

    def initial_round_robin(self):
        while not all(self.future_visits_per_comb_per_arm[(False, False)]):
            arm = self.current_round % self.n_arms
            strategy = {comb: arm for comb in self.env.get_features_combinations()}
            self.pull_from_env(strategy)

        self.performed_round_robins = 4
        self.state_is_explorative_rounds = False
        self.remaining_normal_rounds = 2 ** self.performed_round_robins

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

        self.strategies.append(strategy)
        self.update_round_count()

    def update_round_count(self):
        self.current_round += 1

        if self.state_is_explorative_rounds:
            self.next_round_robin_arm = (self.next_round_robin_arm + 1) % self.n_arms

            if not self.next_round_robin_arm:  # one complete round robin has just ended
                self.remaining_round_robins -= 1

            if not self.remaining_round_robins:
                self.performed_round_robins += 1
                self.state_is_explorative_rounds = False
                self.remaining_normal_rounds = 2 ** self.performed_round_robins
        else:
            self.remaining_normal_rounds -= 1
            if not self.remaining_normal_rounds:
                self.state_is_explorative_rounds = True
                self.remaining_round_robins = self.round_robins_per_cycle
                self.next_round_robin_arm = 0

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

    def compute_cumulative_exp_profits(self, expected_profits):
        return np.cumsum([
            sum(expected_profits[comb][arm] for comb, arm in s.items())
            for s in self.strategies]
        )

    def get_average_conversion_rates(self, context: Context):
        return context.get_average_conversion_rates(self.new_clicks_per_comb_per_arm, self.purchases_per_comb_per_arm)

    def get_context_number_of_pulls(self, context: Context):
        return context.get_number_of_pulls(self.new_clicks_per_comb_per_arm)
