import numpy as np

from src.bandit.Context import Context
from src.distributions import Beta


class TSContext(Context):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # start projection
    def compute_projection_conversion_rate(self, new_clicks_per_arm,
                                           purchases_per_arm, current_round):
        tot_new_clicks_per_arm = [sum(r) for r in new_clicks_per_arm]
        successes_per_arm = [sum(r) for r in purchases_per_arm]
        failures_per_arm = [tot_new_clicks_per_arm[a] - successes_per_arm[a]
                            for a in range(self.n_arms)]

        betas = [Beta(1 + successes_per_arm[a], 1 + failures_per_arm[a], self.rng)
                 for a in range(self.n_arms)]

        crs = np.array([b.sample() for b in betas]).flatten()

        return crs

    # end projection

    def compute_conversion_rate_lower_bounds(self, new_clicks_per_arm, purchases_per_arm, current_round):
        return self._compute_cr_lower_bounds(new_clicks_per_arm, purchases_per_arm, current_round)

    def compute_average_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        return self._compute_cr_averages(new_clicks_per_arm, purchases_per_arm)

    def _compute_cr_averages(self, new_clicks_per_arm, purchases_per_arm):
        return np.array([sum(purchases_per_arm[arm]) / sum(new_clicks_per_arm[arm])
                         for arm in range(self.n_arms)]).flatten()

    def compute_conversion_rates_radia(self, new_clicks_per_arm, current_round):
        tot_clicks_per_arm = np.array([np.sum(new_clicks_per_arm[arm])
                                       for arm in range(self.n_arms)])

        return np.sqrt(2 * np.log(current_round) / tot_clicks_per_arm)

    def _compute_cr_lower_bounds(self, new_clicks_per_arm, purchases_per_arm, current_round):
        averages = self._compute_cr_averages(new_clicks_per_arm, purchases_per_arm)
        radia = self.compute_conversion_rates_radia(new_clicks_per_arm, current_round)
        lower_bounds = averages - radia

        return lower_bounds

    def _get_average_conversion_rates(self, new_clicks_per_arm, purchases_per_arm):
        return self._compute_cr_averages(new_clicks_per_arm, purchases_per_arm)
