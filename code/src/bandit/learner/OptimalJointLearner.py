import numpy as np
from scipy.stats import norm

from src.algorithms import simple_class_profit
from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment
from src.utils import sum_ragged_matrix, average_ragged_matrix


class OptimalJointLearner:
    def __init__(self, env: JointBanditEnvironment):
        self.env = env
        self.n_arms_price = self.env.n_arms_price
        self.n_arms_bid = self.env.n_arms_bid

        self.future_visits = [[[] for i in range(self.n_arms_bid)]
                              for j in range(self.n_arms_price)]
        self.purchases = [[[] for i in range(self.n_arms_bid)]
                          for j in range(self.n_arms_price)]
        self.new_clicks = [[[] for i in range(self.n_arms_bid)]
                           for j in range(self.n_arms_price)]
        self.tot_cost_per_bid = [0 for i in range(self.n_arms_bid)]
        self.tot_auctions_per_bid = [[0] for i in range(self.n_arms_bid)]
        self.current_round = 0
        self.security = 0.2

        self.pulled_arms = []

    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        arm_price, arm_bid = self.choose_next_arm()
        self.pull_from_env(arm_price, arm_bid)

    # start next arm choice

    def choose_next_arm(self):
        median_bid = self.n_arms_bid // 2
        arm_price = np.argmax(self.compute_projected_profits_fixed_bid(median_bid))
        mask = self.compute_safe_arms(arm_price)
        arms_bid_safe = np.where(mask,
                                 self.compute_projected_profits_fixed_price(arm_price), 0)
        arm_bid = np.argmax(arms_bid_safe)
        return arm_price, arm_bid

    # end next arm choice

    def compute_safe_arms(self, arm_price):

        new_c = [[] for b in range(self.n_arms_bid)]
        for b in range(self.n_arms_bid):
            for p in range(self.n_arms_price):
                new_c[b].extend(self.new_clicks[p][b])

        means = [np.mean(new_c[b]) for b in range(self.n_arms_bid)]
        std_devs = [np.std(new_c[b]) for b in range(self.n_arms_bid)]

        lower_security_value = [norm.ppf(self.security, m, std) for m, std in zip(means, std_devs)]

        expected_profits = self.compute_expected_profits_fixed_price(arm_price, nc=lower_security_value)
        arm_mask = expected_profits > 0

        return arm_mask

    # start projected profits fixed bid

    def compute_projected_profits_fixed_bid(self, arm_bid):
        new_clicks = self.compute_new_clicks(arm_bid)
        margin = np.array([self.env.margin(a) for a in range(self.n_arms_price)])
        crs = self.compute_projection_conversion_rates()
        future_visits = self.compute_future_visits_per_arm()
        cost_per_click = self.compute_cost_per_click(arm_bid)

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    # end projected profits fixed bid
    # start projected profits fixed price

    def compute_projected_profits_fixed_price(self, arm_price):
        new_clicks = self.compute_projection_new_clicks()
        margin = self.env.margin(arm_price)
        crs = self.compute_conversion_rates(arm_price)
        future_visits = self.compute_future_visits(arm_price)
        cost_per_click = self.compute_cost_per_click_per_arm()

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    # end projected profits fixed price

    def compute_expected_profits_fixed_price(self, arm_price, nc):
        new_clicks = nc
        margin = self.env.margin(arm_price)
        crs = self.compute_conversion_rates(arm_price)
        future_visits = self.compute_future_visits(arm_price)
        cost_per_click = self.compute_cost_per_click_per_arm()

        expected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return expected_profit

    # start axis projected quantities

    def compute_conversion_rates(self, arm_price):
        return sum_ragged_matrix(self.purchases[arm_price]) / \
               sum_ragged_matrix(self.new_clicks[arm_price])

    def compute_new_clicks(self, arm_bid):
        return average_ragged_matrix([self.new_clicks[p][arm_bid]
                                      for p in range(self.n_arms_price)])

    def compute_cost_per_click(self, arm_bid):
        tot_clicks = np.sum([np.sum(self.new_clicks[arm_p][arm_bid])
                             for arm_p in range(self.n_arms_price)])
        tot_cost = self.tot_cost_per_bid[arm_bid]
        cost_per_click = tot_cost / tot_clicks

        return cost_per_click

    def compute_future_visits_per_arm(self):
        return np.array([self.compute_future_visits(arm_p)
                         for arm_p in range(self.n_arms_price)])

    def compute_future_visits(self, arm_price):
        arm_p_future_visits = 0
        arm_p_purchases = 0
        for arm_b in range(self.n_arms_bid):
            complete_samples = len(self.future_visits[arm_price][arm_b])
            future_visits = np.sum(self.future_visits[arm_price][arm_b])
            purchases = np.sum(self.purchases[arm_price][arm_b][:complete_samples])
            arm_p_future_visits += future_visits
            arm_p_purchases += purchases

        return arm_p_future_visits / arm_p_purchases if arm_p_purchases else 0

    # end axis projected quantities
    # start random variables computation

    def compute_projection_conversion_rates(self):
        raise NotImplementedError

    def compute_projection_auction_winning_probability(self):
        raise NotImplementedError

    def compute_projection_new_clicks(self):
        auction_win_probability = self.compute_projection_auction_winning_probability()
        average_auctions = np.sum(self.tot_auctions_per_bid) / self.current_round

        return auction_win_probability * average_auctions

    # end random variables computation

    def compute_cost_per_click_per_arm(self):
        return np.array([self.compute_cost_per_click(arm_b) for arm_b in range(self.n_arms_bid)])

    def round_robin(self):
        # Just trust Jacopo for this one-liner, I do, you will.
        while not all(any(x) for x in self.future_visits) or any(
                all(not self.new_clicks[p][b] for p in range(self.n_arms_price)) for b in range(self.n_arms_bid)):
            arm_p = self.current_round % self.n_arms_price
            arm_b = self.current_round % self.n_arms_bid
            self.pull_from_env(arm_p, arm_b)

    def pull_from_env(self, arm_price: int, arm_bid: int):
        auctions, new_clicks, purchases, tot_cost, \
        (old_a, visits) = self.env.pull_arm_not_discriminating(arm_price, arm_bid)

        self.tot_auctions_per_bid[arm_bid] += auctions

        self.new_clicks[arm_price][arm_bid].append(new_clicks)
        self.purchases[arm_price][arm_bid].append(purchases)
        self.tot_cost_per_bid[arm_bid] += tot_cost

        if old_a[0] is not None:
            arm_p, arm_b = old_a
            self.future_visits[arm_p][arm_b].append(visits)

        self.current_round += 1
        self.pulled_arms.append((arm_price, arm_bid))

    def pulled_arm_count(self, arm_p, arm_b):
        i = 0
        for arm_price, arm_bid in self.pulled_arms:
            if arm_price == arm_p and arm_bid == arm_b:
                i += 1

        return i

    def get_pulled_arms_recap(self):
        recap = []
        for arm_p in range(self.n_arms_price):
            bid_recap = []
            for arm_b in range(self.n_arms_bid):
                bid_recap.append(self.pulled_arm_count(arm_p, arm_b))
            recap.append(bid_recap)

        return recap

    def compute_cumulative_exp_profits(self, expected_profits):
        return np.cumsum([expected_profits[p][b]
                          for p, b in self.pulled_arms])
