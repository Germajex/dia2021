import numpy as np

from src.algorithms import simple_class_profit
from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.utils import sum_ragged_matrix, average_ragged_matrix


class OptimalPriceLearner:
    def __init__(self, env: PriceBanditEnvironment):
        self.env = env
        self.n_arms = self.env.n_arms
        self.future_visits_per_arm = [[] for i in range(self.n_arms)]
        self.purchases_per_arm = [[] for i in range(self.n_arms)]
        self.new_clicks_per_arm = [[] for i in range(self.n_arms)]
        self.tot_cost = 0
        self.current_round = 0

        self.pulled_arms = []

    # start learning loop
    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        arm = self.choose_next_arm()
        self.pull_from_env(arm=arm)

    def choose_next_arm(self):
        return int(np.argmax(self.compute_projected_profits()))

    # end learning loop
    # start projected profits
    def compute_projected_profits(self):
        average_new_clicks = self.compute_average_new_clicks()
        margin = np.array([self.env.margin(a) for a in range(self.n_arms)])
        crs = self.compute_projection_conversion_rates()
        future_visits = self.compute_future_visits_per_arm()
        tot_clicks = sum_ragged_matrix(self.new_clicks_per_arm)
        cost_per_click = self.tot_cost / tot_clicks

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=average_new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    # end projected profits

    def compute_expected_profits(self):
        average_new_clicks = self.compute_average_new_clicks()
        margin = np.array([self.env.margin(a) for a in range(self.n_arms)])
        crs = self.get_average_conversion_rates()
        future_visits = self.compute_future_visits_per_arm()
        tot_clicks = sum_ragged_matrix(self.new_clicks_per_arm)

        cost_per_click = self.tot_cost / tot_clicks

        expected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=average_new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return expected_profit

    # start compute estimates
    def compute_future_visits_per_arm(self):
        res = []
        for arm in range(self.n_arms):
            complete_samples = len(self.future_visits_per_arm[arm])
            future_visits = np.sum(self.future_visits_per_arm[arm])
            purchases = np.sum(self.purchases_per_arm[arm][:complete_samples])
            future_visits_per_purchase = future_visits / purchases if purchases else 0
            res.append(future_visits_per_purchase)

        return np.array(res)

    def compute_average_new_clicks(self):
        return average_ragged_matrix(self.new_clicks_per_arm)

    # end compute estimates
    # start conversion rates
    def compute_projection_conversion_rates(self):
        raise NotImplementedError

    def get_average_conversion_rates(self):
        raise NotImplementedError

    # end conversion rates

    def get_number_of_pulls(self):
        return [len(a) for a in self.new_clicks_per_arm]

    def round_robin(self):
        while not all(self.future_visits_per_arm):
            arm = self.current_round % self.n_arms
            self.pull_from_env(arm)

    def pull_from_env(self, arm: int):
        new_clicks, purchases, tot_cost, \
            (old_a, visits) = self.env.pull_arm_not_discriminating(arm)

        self.new_clicks_per_arm[arm].append(new_clicks)
        self.purchases_per_arm[arm].append(purchases)
        self.tot_cost += tot_cost

        if old_a is not None:
            self.future_visits_per_arm[old_a].append(visits)

        self.current_round += 1
        self.pulled_arms.append(arm)
        
        # return value used in TSOptimalPriceLearner
        return new_clicks, purchases, tot_cost, (old_a, visits)

    def compute_cumulative_exp_profits(self, expected_profits):
        return np.cumsum([expected_profits[a] for a in self.pulled_arms])

    def compute_cumulative_regr_from_gaps(self, gaps):
        return np.cumsum([gaps[a] for a in self.pulled_arms])
