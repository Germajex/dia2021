from typing import List

import numpy as np
from src.bandit.BanditEnvironment import BanditEnvironment


class OptimalPriceLearner:
    def __init__(self, env: BanditEnvironment):
        self.env = env
        self.n_arms = self.env.n_arms
        self.future_visits_per_arm = [[] for i in range(self.n_arms)]
        self.purchases_per_arm = [[] for i in range(self.n_arms)]
        self.new_clicks_per_arm: List[List[int]] = [[] for i in range(self.n_arms)]
        self.tot_cost_per_click = 0
        self.current_round = 0

    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()
            #print(f'{self.current_round}')

    def learn_one_round(self):
        arm = self.choose_next_arm()
        self.pull_from_env(arm=arm)

    def choose_next_arm(self):
        return int(np.argmax(self.compute_projected_profits()))

    def compute_projected_profits(self):
        new_clicks = self.compute_new_clicks()
        margin = np.array([self.env.margin(a) for a in range(self.n_arms)])
        crs = self.compute_projection_conversion_rates()
        future_visits = self.compute_future_visits()
        cost_per_click = self.tot_cost_per_click / self.sum_ragged_matrix(self.new_clicks_per_arm)

        projected_profit = new_clicks * (margin * crs * (1 + future_visits) - cost_per_click)

        return projected_profit

    def compute_expected_profits(self):
        new_clicks = self.compute_new_clicks()
        margin = np.array([self.env.margin(a) for a in range(self.n_arms)])
        crs = self.get_average_conversion_rates()
        future_visits = self.compute_future_visits()
        cost_per_click = self.tot_cost_per_click / self.sum_ragged_matrix(self.new_clicks_per_arm)

        projected_profit = new_clicks * (margin * crs * (1 + future_visits) - cost_per_click)

        return projected_profit

    def compute_future_visits(self):
        successes_sum = 0

        for arm in range(self.n_arms):
            complete_samples = len(self.future_visits_per_arm[arm])
            successes_sum += np.sum(self.purchases_per_arm[arm][:complete_samples])

        return self.sum_ragged_matrix(self.future_visits_per_arm)/successes_sum

    def compute_new_clicks(self):
        return self.average_ragged_matrix(self.new_clicks_per_arm)

    def compute_projection_conversion_rates(self):
        raise NotImplementedError

    def get_average_conversion_rates(self):
        raise NotImplementedError

    def get_number_of_pulls(self):
        return [len(a) for a in self.new_clicks_per_arm]

    def round_robin(self):
        while not all(self.future_visits_per_arm):
            arm = self.current_round % self.n_arms
            self.pull_from_env(arm)

    def pull_from_env(self, arm: int):
        new_clicks, purchases, tot_cost_per_clicks, \
        (old_a, visits) = self.env.pull_arm_not_discriminating(arm)

        self.new_clicks_per_arm[arm].append(new_clicks)
        self.purchases_per_arm[arm].append(purchases)
        self.tot_cost_per_click += tot_cost_per_clicks

        if old_a is not None:
            self.future_visits_per_arm[old_a].append(visits)

        self.current_round += 1

        return new_clicks, purchases, tot_cost_per_clicks, (old_a, visits)

    @staticmethod
    def average_ragged_matrix(mat):
        return OptimalPriceLearner.sum_ragged_matrix(mat) / OptimalPriceLearner.count_ragged_matrix(mat)

    @staticmethod
    def count_ragged_matrix(mat):
        return np.sum(np.fromiter((len(r) for r in mat), dtype=np.int32))

    @staticmethod
    def sum_ragged_matrix(mat: List[List[int]]):
        return np.sum(np.fromiter((np.sum(r) for r in mat), dtype=np.float64))

    def get_cumulative_rewards(self):
        return np.cumsum(self.history_rewards)

    # This method runs a first round of exploration, initializing the UCB algorithm
    def explore_price(self, mode):
        for a in range(self.n_arms):
            self.pull_arm_price(a, mode)

    # The 'core' UCB algorithm, it learns the best price possible, given the number of rounds, the possible prices
    # and the fixed bid
    def learn_price(self, n_rounds, prices, bid, mode='cr', c_param=None, verbose=True):
        if mode != 'cr' and mode != 'rwd':
            print(f"[error] !!! learning mode {mode} is not available !!!")
            return

        # Set hyper-parameter c for confidence bound computation
        if c_param is not None:
            self.c = c_param
        elif mode == 'cr':
            self.c = 0.1
        elif mode == 'rwd':
            self.c = 0.01
        else:
            self.c = 1

        self.prices = prices
        self.bids = [bid]

        self.explore_price(mode)

        for i in range(self.n_arms, n_rounds):
            a = self.choose_arm(i, mode)
            self.pull_arm_price(a, mode)

            if mode == 'cr' and i % 10 == 0:
                if verbose:
                    pass
                    # self.wait_and_show(i, mode)

        if verbose:
            self.pulled_arms_recap(mode)

    # Choose the arm to pull based on the reward upper bound (reward computed with cr upper bound).
    # Returns the index of the arm to be pulled
    def choose_arm(self, t, mode):
        # The choice of the arm to be pulled is based on the
        if mode == 'rwd':
            rwds_array = [self.ucb_upper_bound(a, t, mode) for a in range(self.n_arms)]
            return np.argmax(rwds_array)

        elif mode == 'cr':
            cr_upper_bound_array = [self.ucb_upper_bound(a, t, mode) for a in range(self.n_arms)]
            rwds_array = [self.env.get_total_reward(a, self.bids[0], cr) for a, cr in
                          zip(self.prices, cr_upper_bound_array)]
            return np.argmax(rwds_array)

    # Computes the ucb upper bound of a given arm at a given time
    def ucb_upper_bound(self, arm, time, mode):
        value_array = []
        if mode == 'cr':
            value_array = self.cr_per_arm
        elif mode == 'rwd':
            value_array = self.rewards_per_arm

        return np.mean(value_array[arm]) + self.ucb_confidence_bound(arm, time, mode)

    # Computes the ucb lower bound of a given arm at a given time
    def ucb_lower_bound(self, arm, time, mode):
        value_array = []
        if mode == 'cr':
            value_array = self.cr_per_arm
        elif mode == 'rwd':
            value_array = self.rewards_per_arm

        return np.mean(value_array[arm]) - self.ucb_confidence_bound(arm, time, mode)

    # Implementation of the UCB confidence bound. It needs the arm, the timestamp and an hyper-parameter c
    def ucb_confidence_bound(self, a, t, mode):
        if mode == 'cr':
            n_pulls = len(self.cr_per_arm[a])
            return self.c * np.sqrt(2 * np.log(t) / n_pulls)
        elif mode == 'rwd':
            n_pulls = len(self.rewards_per_arm[a])
            b = self.get_max_reward()
            sigma = 1/(t**4)
            num = self.c*np.log((4*self.n_arms*(t**3))/sigma)
            den = n_pulls

            return b*np.sqrt(num/den)

    def get_max_reward(self):
        max_rwd = 0
        for r in self.rewards_per_arm:
            r_max = np.max(r)
            if r_max > max_rwd:
                max_rwd = r_max

        return max_rwd

    # Prints out a brief recap of the activities done by the learner
    def pulled_arms_recap(self, mode):
        print(f"\nUCB Learner working in mode {mode}")

        for a in range(self.n_arms):
            n_pulls = len(self.rewards_per_arm[a])
            avg_reward = np.mean(self.rewards_per_arm[a])
            avg_cr = np.mean(self.cr_per_arm[a])

            print(
                f"[info] Arm {a} pulled {n_pulls} times - avg reward: {avg_reward:.2f} - avg cr: {avg_cr:.2f}")
