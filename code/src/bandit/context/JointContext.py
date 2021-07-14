import numpy as np
from scipy.stats import norm

from src.algorithms import simple_class_profit
from src.utils import sum_ragged_matrix, average_ragged_matrix


class JointContext:
    def __init__(self, features, arm_margin_function, n_arms_price, n_arms_bid, rng):
        self.features = features
        self.margin = arm_margin_function
        self.n_arms_price = n_arms_price
        self.n_arms_bid = n_arms_bid
        self.rng = rng

        # Arms merged data
        self.future_visits = None
        self.purchases = None
        self.new_clicks = None
        self.tot_cost_per_bid = None
        self.tot_auctions_per_bid = None
        self.pulled_arms = []

    # start merge

    def merge_all_data(self, future_visits_per_comb, purchases_per_comb,
                       new_clicks_per_comb, tot_cost_per_comb, tot_auctions_per_comb):

        self.future_visits = self.merge_double_indexed(future_visits_per_comb)
        self.purchases = self.merge_double_indexed(purchases_per_comb)
        self.new_clicks = self.merge_double_indexed(new_clicks_per_comb)
        self.tot_cost_per_bid = self.merge_single_indexed_bid(tot_cost_per_comb)
        self.tot_auctions_per_bid = self.merge_single_indexed_bid(tot_auctions_per_comb)

    def merge_double_indexed(self, data_per_comb):
        data = []
        # merged_data[price_arm][price_bid][realization_n]
        merged_data = [[[]
                        for b in range(self.n_arms_bid)]
                       for p in range(self.n_arms_price)]

        for comb in self.features:
            row = data_per_comb[comb]
            data.append(row)

        # cycle through all prices and bids
        for p in range(self.n_arms_price):
            for b in range(self.n_arms_bid):
                for i in range(len(data[0][p][b])):
                    # j-th combination, i-th realization
                    merged_data[p][b].append(np.sum([data[j][p][b][i]
                                                     for j in range(len(data))]))

        return merged_data

    def merge_single_indexed_bid(self, data_per_comb):
        merged_data = np.sum([data_per_comb[comb] for comb in self.features], axis=0)
        return merged_data

    # end merge
    # start arm choice

    def choose_next_arm(self, security, current_round):
        median_bid = self.n_arms_bid // 2
        arm_price = np.argmax(self.compute_projected_profits_fixed_bid(median_bid,
                                                                       current_round))
        mask = self.compute_safe_arms(arm_price, security)
        arms_bid_safe = np.where(mask,
                                 self.compute_projected_profits_fixed_price(arm_price,
                                                                            current_round),
                                 0)
        arm_bid = np.argmax(arms_bid_safe)
        return arm_price, arm_bid

    def compute_safe_arms(self, arm_price, security):
        new_c = [[] for i in range(self.n_arms_bid)]
        for b in range(self.n_arms_bid):
            for p in range(self.n_arms_price):
                new_c[b].extend(self.new_clicks[p][b])

        means = np.array([np.mean(new_c[b]) for b in range(self.n_arms_bid)])
        std_devs = np.array([np.std(new_c[b]) for b in range(self.n_arms_bid)])
        std_devs = np.where(std_devs > 0, std_devs, 1)

        lower_security_value = np.array([norm.ppf(security, m, std)
                                         for m, std in zip(means, std_devs)])
        lower_security_value = np.where(lower_security_value > 0, lower_security_value, 0)

        expected_profits = self.compute_expected_profits_fixed_price(arm_price,
                                                                     lower_security_value)
        arm_mask = expected_profits > 0

        return arm_mask

    # end arm choice

    def compute_projected_profits_fixed_bid(self, arm_bid, current_round):
        new_clicks = self.compute_new_clicks(arm_bid)
        margin = np.array([self.margin(a) for a in range(self.n_arms_price)])
        crs = self.compute_projection_conversion_rates(current_round)
        future_visits = self.compute_future_visits_per_arm()
        cost_per_click = self.compute_cost_per_click(arm_bid)

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    def compute_projected_profits_fixed_price(self, arm_price, current_round):
        new_clicks = self.compute_projection_new_clicks(current_round)
        margin = self.margin(arm_price)
        crs = self.compute_conversion_rates(arm_price)
        future_visits = self.compute_future_visits(arm_price)
        cost_per_click = self.compute_cost_per_click_per_arm()

        projected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return projected_profit

    def compute_expected_profits_fixed_price(self, arm_price, nc):
        new_clicks = nc
        margin = self.margin(arm_price)
        crs = self.compute_conversion_rates(arm_price)
        future_visits = self.compute_future_visits(arm_price)
        cost_per_click = self.compute_cost_per_click_per_arm()

        expected_profit = simple_class_profit(
            margin=margin, conversion_rate=crs, new_clicks=new_clicks,
            future_visits=future_visits, cost_per_click=cost_per_click
        )

        return expected_profit

    def compute_conversion_rates(self, arm_price):
        return sum_ragged_matrix(self.purchases[arm_price]) / sum_ragged_matrix(self.new_clicks[arm_price])

    def compute_new_clicks(self, arm_bid):
        return average_ragged_matrix([self.new_clicks[p][arm_bid] for p in range(self.n_arms_price)])

    def compute_future_visits_per_arm(self):
        return np.array([self.compute_future_visits(arm_p) for arm_p in range(self.n_arms_price)])

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

    def compute_projection_conversion_rates(self, current_round):
        raise NotImplementedError

    def compute_projection_new_clicks(self, current_round):
        auction_win_probability = self.compute_projection_auction_winning_probability(current_round)
        average_auctions = np.sum(self.tot_auctions_per_bid) / current_round

        return auction_win_probability * average_auctions

    def compute_projection_auction_winning_probability(self, current_round):
        raise NotImplementedError

    def compute_cost_per_click_per_arm(self):
        return np.array([self.compute_cost_per_click(arm_b) for arm_b in range(self.n_arms_bid)])

    def compute_cost_per_click(self, arm_bid):
        tot_clicks = np.sum([np.sum(self.new_clicks[arm_p][arm_bid]) for arm_p in range(self.n_arms_price)])
        tot_cost = self.tot_cost_per_bid[arm_bid]

        if tot_clicks == 0:
            tot_clicks += 1

        cost_per_click = tot_cost / tot_clicks

        return cost_per_click

    def update_pulled_arms(self, strategy_price, strategy_bid):
        price = strategy_price[self.features[0]]
        bid = strategy_bid[self.features[0]]
        self.pulled_arms.append((price, bid))

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
