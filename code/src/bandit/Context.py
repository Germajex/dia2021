import numpy as np

from src.utils import sum_ragged_matrix, average_ragged_matrix


class Context:
    def __init__(self, features, arm_margin_function, n_arms):
        self.features = features
        self.margin = arm_margin_function
        self.n_arms = n_arms

        # caso problematico: context che gestisce combinazioni A, B riceve il feedback di un pull
        # fatto quando A e B erano splittati, e per i due sono stati pullati arm diversi.
        # come piazzare le future visits? quindi abbiamo diviso tutto per comb
        # e dato che l'environment ha già tutto diviso per comb semplicemente le passa come parametri
        # e il context se li calcola aggregati

    def choose_next_arm(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm, cost_per_click_per_comb,
                        future_visits_per_comb_per_arm, current_round):
        return int(np.argmax(self.compute_projected_profits(new_clicks_per_comb_per_arm, purchases_per_comb_per_arm,
                                                            cost_per_click_per_comb, future_visits_per_comb_per_arm,
                                                            current_round)))

    def merge_cost_per_click(self, cost_per_click_per_comb):
        cost_per_click = sum(cost_per_click_per_comb[comb] for comb in self.features)
        return cost_per_click

    def merge(self, data_per_comb_per_arm):
        data_per_arm = [
            [
                sum(data_per_comb_per_arm[comb][arm][r] for comb in self.features)
                for r in range(len(data_per_comb_per_arm[self.features[0]][arm]))
            ]

            for arm in range(self.n_arms)
        ]
        return data_per_arm

    def compute_projected_profits(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm,
                                  cost_per_click_per_comb, future_visits_per_comb_per_arm, current_round):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)
        tot_cost_per_click = self.merge_cost_per_click(cost_per_click_per_comb)

        future_visits_per_arm = self.merge(future_visits_per_comb_per_arm)

        average_new_clicks = self.compute_average_new_clicks(new_clicks_per_arm)

        margin = np.array([self.margin(a) for a in range(self.n_arms)])

        crs = self.compute_projection_conversion_rates(new_clicks_per_arm, purchases_per_arm, current_round)

        future_visits_per_arm = self.compute_future_visits(future_visits_per_arm, purchases_per_arm)
        cost_per_click = tot_cost_per_click / sum_ragged_matrix(new_clicks_per_arm)

        projected_profit = average_new_clicks * (margin * crs * (1 + future_visits_per_arm) - cost_per_click)

        return projected_profit

    def compute_expected_profits(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm, cost_per_click_per_comb,
                                 future_visits_per_comb_per_arm):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)
        tot_cost_per_click = self.merge_cost_per_click(cost_per_click_per_comb)

        future_visits_per_arm = self.merge(future_visits_per_comb_per_arm)

        average_new_clicks = self.compute_average_new_clicks(new_clicks_per_arm)

        margin = np.array([self.margin(a) for a in range(self.n_arms)])

        crs = self.compute_expected_conversion_rates(new_clicks_per_arm, purchases_per_arm)

        future_visits_per_arm = self.compute_future_visits(future_visits_per_arm, purchases_per_arm)
        cost_per_click = tot_cost_per_click / sum_ragged_matrix(new_clicks_per_arm)

        projected_profit = average_new_clicks * (margin * crs * (1 + future_visits_per_arm) - cost_per_click)

        return projected_profit

    def compute_purchases(self, pur):
        return sum(sum_ragged_matrix(pur[comb] for comb in self.features))

    def compute_average_new_clicks(self, new_clicks):
        average_new_clicks = average_ragged_matrix(new_clicks)
        return average_new_clicks

    def compute_projection_conversion_rates(self, new_clicks_per_arm, purchases_per_arm, current_round):
        raise NotImplementedError

    def compute_expected_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        raise NotImplementedError

    def compute_future_visits(self, future_visits_per_arm, purchases_per_arm):
        successes_sum = 0

        for arm in range(self.n_arms):
            complete_samples = len(future_visits_per_arm[arm])
            successes_sum += np.sum(purchases_per_arm[arm][:complete_samples])

        return sum_ragged_matrix(future_visits_per_arm) / successes_sum

    def get_average_conversion_rates(self, new_clicks_per_comb_per_arm, purchases_per_comb_per_arm):
        new_clicks_per_arm = self.merge(new_clicks_per_comb_per_arm)
        purchases_per_arm = self.merge(purchases_per_comb_per_arm)

        return self._get_average_conversion_rates(new_clicks_per_arm, purchases_per_arm)

    def _get_average_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        raise NotImplementedError
