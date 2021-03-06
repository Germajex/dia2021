import numpy as np

from src.algorithms import simple_class_profit
from src.bandit.banditEnvironments import BidBanditEnvironment
from src.utils import average_ragged_matrix, sum_ragged_matrix
from scipy.stats import norm


class OptimalBidLearner:
    def __init__(self, env: BidBanditEnvironment):
        self.env = env
        self.n_arms = self.env.n_arms
        self.future_visits_per_arm = [[] for i in range(self.n_arms)]
        self.purchases_per_arm = [[] for i in range(self.n_arms)]
        self.auctions_per_arm = [[] for i in range(self.n_arms)]
        self.new_clicks_per_arm = [[] for i in range(self.n_arms)]
        self.tot_cost_per_arm = [0 for i in range(self.n_arms)]
        self.current_round = 0
        self.pulled_arms = []
        self.security = 0.2

    # start learning loop
    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        arm = self.choose_next_arm()
        self.pull_from_env(arm=arm)

    def choose_next_arm(self):
        mask = self.compute_safe_arms()
        # put compute_projected_profits where mask is true and 0 otherwise
        arms_bid_safe = np.where(mask, self.compute_projected_profits(), 0)
        return int(np.argmax(arms_bid_safe))

    # end learning loop

    # start safe arms
    def compute_safe_arms(self):
        means = [np.mean(new_clicks) for new_clicks in self.new_clicks_per_arm]
        std_dev = [np.std(new_clicks) for new_clicks in self.new_clicks_per_arm]

        lower_security_value = [norm.ppf(self.security, m, std)
                                for m, std in zip(means, std_dev)]
        expected_profits = self.compute_expected_profits(nc=lower_security_value)
        arm_mask = expected_profits > 0

        return arm_mask

    # end safe arms

    # start projected profits
    def compute_projected_profits(self):
        auctions = self.compute_average_auctions()
        winning_probs = self.compute_projection_auction_winning_probability_per_arm()
        new_clicks = auctions * winning_probs
        # constant margin of the fixed price
        margin = self.env.margin()
        crs = self.compute_conversion_rates()
        future_visits = self.compute_future_visits()
        cost_per_click = np.array(
            [self.tot_cost_per_arm[arm] / np.sum(self.new_clicks_per_arm[arm])
             for arm in range(self.n_arms)]
        )

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    # end projected profit

    def compute_expected_profits(self, nc=None):
        new_clicks = nc
        if nc is None:
            auctions = self.compute_average_auctions()
            win_probs = self.compute_average_auction_winning_probability_per_arm()
            new_clicks = auctions * win_probs

        margin = self.env.margin()
        crs = self.compute_conversion_rates()
        future_visits = self.compute_future_visits()
        cost_per_click = np.array(
            [self.tot_cost_per_arm[arm] / np.sum(self.new_clicks_per_arm[arm])
             for arm in range(self.n_arms)])

        expected_profit = new_clicks * (margin * crs * (1 + future_visits) - cost_per_click)

        return expected_profit

    def compute_average_new_clicks_per_arm(self):
        average_clicks_per_arm = np.array(
            [np.sum(np.array(self.new_clicks_per_arm[arm]) / len(self.new_clicks_per_arm[arm]))
             for arm in range(self.n_arms)])

        return average_clicks_per_arm

    def compute_average_auction_winning_probability_per_arm(self):
        avgs = [sum(self.new_clicks_per_arm[arm]) / sum(self.auctions_per_arm[arm])
                for arm in range(self.n_arms)]

        return np.array(avgs)

    def compute_projection_auction_winning_probability_per_arm(self):
        raise NotImplementedError

    # start estimated quantities
    def compute_average_auctions(self):
        return average_ragged_matrix(self.auctions_per_arm)

    def compute_conversion_rates(self):
        return sum_ragged_matrix(self.purchases_per_arm) / \
               sum_ragged_matrix(self.new_clicks_per_arm)

    def compute_future_visits(self):
        successes_sum = 0

        for arm in range(self.n_arms):
            complete_samples = len(self.future_visits_per_arm[arm])
            successes_sum += np.sum(self.purchases_per_arm[arm][:complete_samples])

        if not successes_sum:
            return 0
        else:
            return sum_ragged_matrix(self.future_visits_per_arm) / successes_sum

    # end estimated quantities

    def get_number_of_pulls(self):
        return np.array([len(a) for a in self.new_clicks_per_arm])

    def round_robin(self):
        while not all(self.future_visits_per_arm):
            arm = self.current_round % self.n_arms
            self.pull_from_env(arm)

    def pull_from_env(self, arm: int):
        auctions, new_clicks, purchases, tot_cost, \
            (old_a, visits) = self.env.pull_arm_not_discriminating(arm)

        self.new_clicks_per_arm[arm].append(new_clicks)
        self.auctions_per_arm[arm].append(auctions)
        self.purchases_per_arm[arm].append(purchases)
        self.tot_cost_per_arm[arm] += tot_cost

        if old_a is not None:
            self.future_visits_per_arm[old_a].append(visits)

        self.current_round += 1
        self.pulled_arms.append(arm)

    def compute_cumulative_exp_profits(self, expected_profits):
        return np.cumsum([expected_profits[a] for a in self.pulled_arms])
