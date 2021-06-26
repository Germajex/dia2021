import numpy as np

from src.bandit.Context import Context


class UCBContext(Context):
    def compute_projection_conversion_rates(self, new_clicks_per_arm, purchases_per_arm, current_round):
        return self.compute_conversion_rates_upper_bounds(new_clicks_per_arm, purchases_per_arm, current_round)

    def compute_expected_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        return self.compute_conversion_rates_averages(new_clicks_per_arm, purchases_per_arm)

    def compute_conversion_rates_averages(self, new_clicks_per_arm, purchases_per_arm):
        return np.array([sum(purchases_per_arm[arm]) / sum(new_clicks_per_arm[arm])
                         for arm in range(self.n_arms)]).flatten()

    def compute_conversion_rates_radia(self, new_clicks_per_arm, current_round):
        tot_clicks_per_arm = np.array([np.sum(new_clicks_per_arm[arm])
                                       for arm in range(self.n_arms)])

        return np.sqrt(2 * np.log(current_round) / tot_clicks_per_arm)

    def compute_conversion_rates_upper_bounds(self, new_clicks_per_arm, purchases_per_arm, current_round):
        averages = self.compute_conversion_rates_averages(new_clicks_per_arm, purchases_per_arm)
        radia = self.compute_conversion_rates_radia(new_clicks_per_arm, current_round)
        upper_bounds = averages + radia

        return upper_bounds

    def _get_average_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        return self.compute_conversion_rates_averages(new_clicks_per_arm, purchases_per_arm)
