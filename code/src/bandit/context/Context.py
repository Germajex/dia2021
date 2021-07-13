import numpy as np

from src.algorithms import simple_class_profit
from src.utils import sum_ragged_matrix, average_ragged_matrix


class Context:
    def __init__(self, features, arm_margin_function, n_arms, rng):
        self.features = features
        self.margin = arm_margin_function
        self.n_arms = n_arms
        self.rng = rng

    # start context next arm
    def choose_next_arm(self, new_clicks_per_comb_per_arm,
                        purchases_per_comb_per_arm, tot_cost_per_comb,
                        future_visits_per_comb_per_arm, current_round):
        return int(np.argmax(self.compute_projected_profit(
            new_clicks_per_comb_per_arm,
            purchases_per_comb_per_arm,
            tot_cost_per_comb,
            future_visits_per_comb_per_arm,
            current_round)
        ))

    # end context next arm

    def merge_cost(self, cost_per_click_per_comb):
        cost_per_click = sum(cost_per_click_per_comb[comb] for comb in self.features)
        return cost_per_click

    def merge(self, data_per_comb_per_arm):
        data_per_arm = []
        for arm in range(self.n_arms):
            lengths = [len(data_per_comb_per_arm[comb][arm]) for comb in self.features]
            row_length = lengths[0]

            assert (len(np.unique(lengths)) == 1)

            new_row = [
                sum(data_per_comb_per_arm[comb][arm][r] for comb in self.features)
                for r in range(row_length)
            ]
            data_per_arm.append(new_row)

        return data_per_arm

    def compute_projected_profit(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm,
                                 tot_cost_per_comb, future_visits_per_comb_per_arm, current_round):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)
        tot_cost = self.merge_cost(tot_cost_per_comb)
        future_visits_per_arm = self.merge(future_visits_per_comb_per_arm)

        average_new_clicks = self.compute_average_new_clicks(new_clicks_per_arm)
        margin = np.array([self.margin(a) for a in range(self.n_arms)])
        crs = self.compute_projection_conversion_rate(new_clicks_per_arm, purchases_per_arm, current_round)
        future_visits_per_purchase_per_arm = self.compute_future_visits_per_purchase_per_arm(future_visits_per_arm,
                                                                                             purchases_per_arm)
        cost_per_click = tot_cost / sum_ragged_matrix(new_clicks_per_arm)

        profits = simple_class_profit(
            margin=margin, new_clicks=average_new_clicks, conversion_rate=crs,
            future_visits=future_visits_per_purchase_per_arm, cost_per_click=cost_per_click
        )

        return profits

    def compute_expected_profit_lower_bound(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm,
                                            tot_cost_per_comb, future_visits_per_comb_per_arm, current_round):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)
        tot_cost = self.merge_cost(tot_cost_per_comb)
        future_visits_per_arm = self.merge(future_visits_per_comb_per_arm)
        average_new_clicks = self.compute_average_new_clicks(new_clicks_per_arm)

        optimal_arm = np.argmax(self.compute_expected_profits(new_clicks_per_comb_per_arm, purchases_per_comb_per_arm,
                                                              tot_cost_per_comb, future_visits_per_comb_per_arm))

        margin = self.margin(optimal_arm)

        cr = self.compute_conversion_rate_lower_bounds(new_clicks_per_arm, purchases_per_arm, current_round)[
            optimal_arm]

        future_visits_per_purchase_per_arm = self.compute_future_visits_per_purchase_per_arm(future_visits_per_arm,
                                                                                             purchases_per_arm)[
            optimal_arm]
        cost_per_click = tot_cost / sum_ragged_matrix(new_clicks_per_arm)

        profit = simple_class_profit(
            margin=margin, new_clicks=average_new_clicks, conversion_rate=cr,
            future_visits=future_visits_per_purchase_per_arm, cost_per_click=cost_per_click
        )

        return profit

    def compute_expected_profits(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm, tot_cost_per_comb,
                                 future_visits_per_comb_per_arm):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)
        tot_cost = self.merge_cost(tot_cost_per_comb)

        future_visits_per_arm = self.merge(future_visits_per_comb_per_arm)

        average_new_clicks = self.compute_average_new_clicks(new_clicks_per_arm)

        margin = np.array([self.margin(a) for a in range(self.n_arms)])

        crs = self.compute_average_conversion_rates(new_clicks_per_arm, purchases_per_arm)

        future_visits_per_purchase_per_arm = self.compute_future_visits_per_purchase_per_arm(future_visits_per_arm,
                                                                                             purchases_per_arm)
        cost_per_click = tot_cost / sum_ragged_matrix(new_clicks_per_arm)

        profits = simple_class_profit(
            margin=margin, new_clicks=average_new_clicks, conversion_rate=crs,
            future_visits=future_visits_per_purchase_per_arm, cost_per_click=cost_per_click
        )

        return profits

    def compute_purchases(self, pur):
        return sum(sum_ragged_matrix(pur[comb] for comb in self.features))

    def compute_average_new_clicks(self, new_clicks):
        average_new_clicks = average_ragged_matrix(new_clicks)
        return average_new_clicks

    def compute_conversion_rate_lower_bounds(self, new_clicks_per_arm, purchases_per_arm, current_round):
        raise NotImplementedError

    def compute_projection_conversion_rate(self, new_clicks_per_arm, purchases_per_arm, current_round):
        raise NotImplementedError

    def compute_average_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        raise NotImplementedError

    def compute_future_visits_per_purchase_per_arm(self, future_visits_per_arm, purchases_per_arm):
        res = []
        for arm in range(self.n_arms):
            complete_samples = len(future_visits_per_arm[arm])
            future_visits = np.sum(future_visits_per_arm[arm])
            purchases = np.sum(purchases_per_arm[arm][:complete_samples])
            future_visits_per_purchase = future_visits / purchases if purchases else 0
            res.append(future_visits_per_purchase)

        return np.array(res)

    def get_average_conversion_rates(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)

        return self._get_average_conversion_rates(new_clicks_per_arm, purchases_per_arm)

    def get_number_of_pulls(self, new_clicks_per_comb_per_arm):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        return [len(r) for r in new_clicks_per_arm]

    def _get_average_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        raise NotImplementedError

"""

    def compute_future_visits_per_purchase(self, future_visits_per_arm, purchases_per_arm):
        purchases_sum = 0

        for arm in range(self.n_arms):
            complete_samples = len(future_visits_per_arm[arm])
            purchases_sum += np.sum(purchases_per_arm[arm][:complete_samples])

        return sum_ragged_matrix(future_visits_per_arm) / purchases_sum

    def compute_expected_profit_last_round(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm,
                                           cost_per_click_per_comb,
                                           future_visits_per_comb_per_arm, arm):

        new_clicks = self.merge(new_clicks_per_comb_per_arm)[arm][-1]
        purchases = self.merge(purchases_per_comb_per_arm)[arm][-1]
        if new_clicks == 0:
            cr = 0
        else:
            cr = purchases / new_clicks
        tot_cost_per_click = self.merge_cost(cost_per_click_per_comb)
        cost_per_click = tot_cost_per_click / sum_ragged_matrix(self.merge(new_clicks_per_comb_per_arm))
        future_visits = self.compute_future_visits_per_purchase(self.merge(future_visits_per_comb_per_arm),
                                                                self.merge(purchases_per_comb_per_arm))
        margin = self.margin(arm)

        expected_profit = new_clicks * (margin * cr * (1 + future_visits) - cost_per_click)
        return expected_profit
"""