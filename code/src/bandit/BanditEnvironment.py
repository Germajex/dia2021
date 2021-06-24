import numpy as np

from src.Environment import Environment
from src.algorithms import optimal_price_for_bid, simple_class_profit
from src.CustomerClass import CustomerClass


# This class is the basic environment for a Bandit learning. Provides many black-box functionalities to the learners
# Allows to set prices, bids, run rounds, get the clairvoyant and compute regret
class BanditEnvironment:
    def __init__(self, environment: Environment, prices, bid, future_visits_delay: int):
        # Init local vars
        self.n_arms = len(prices)
        self.prices = prices
        self.bid = bid
        self.env = environment
        self.future_visits_delay = future_visits_delay

        self.future_visits_queue = None
        self.reset_state()

        self.current_round = 0

    def pull_arm_not_discriminating(self, arm: int):
        new_clicks, purchases, tot_cost_per_clicks, past_future_visits = self._inner_pull_arm(arm)

        return sum(new_clicks), sum(purchases), sum(tot_cost_per_clicks), \
            (past_future_visits[0], sum(past_future_visits[1]))

    def pull_arm_discriminating(self, arm: int):
        print('')

    def _inner_pull_arm(self, arm: int):
        price = self.prices[arm]

        new_clicks = [self.env.distNewClicks.sample(customer_class=c, bid=self.bid)
                      for c in self.env.classes]

        purchases = [self.env.distClickConverted.sample_n(c, price, new_clicks[i])
                     for i, c in enumerate(self.env.classes)]

        tot_future_visits = tuple(
            [sum(self.env.distFutureVisits.sample_n(c, purchases[i]))
             for i, c in enumerate(self.env.classes)]
        )
        self.future_visits_queue.append((arm,  tot_future_visits))

        tot_cost_per_clicks = [
            sum(self.env.distCostPerClick.sample_n(c, self.bid, new_clicks[i]))
            for i, c in enumerate(self.env.classes)
        ]

        past_future_visits = self.future_visits_queue.pop(0)

        self.current_round += 1

        return new_clicks, purchases, tot_cost_per_clicks, past_future_visits

    def reset_state(self):
        self.future_visits_queue = [(None, (0, 0, 0))] * self.future_visits_delay

    def margin(self, arm:int):
        return self.env.margin(self.prices[arm])

    # This methods returns the best reward possible, given a set of possible prices and a fixed bid
    def get_optimal_reward(self, prices, bid):
        optimal_arm = self.get_optimal_arm(prices, bid)
        rwd = np.sum([self.get_class_reward(class_n, prices[optimal_arm], bid, new_clicks_rounded=True)
                      for class_n in range(len(self.env.classes))])
        return rwd

    # This method returns the optimal CR, given a set of prices and a fixed bid
    def get_optimal_cr(self, prices, bid):
        optimal_arm = self.get_optimal_arm(prices, bid)

        total_clicks = np.sum([self.env.distNewClicks.mean(c, bid) for c in self.env.classes])
        successes = np.sum(
            [
                self.env.distClickConverted.mean(c, prices[optimal_arm]) * self.env.distNewClicks.mean(c, bid)
                for c in self.env.classes
            ])

        cr = successes / total_clicks

        return cr

    # This method returns the index of the optimal arm (price). Given a set of prices and a fixed bid
    def get_optimal_arm(self, prices, bid):
        optimal_price = optimal_price_for_bid(self.env, prices=prices, bid=bid)
        return list(prices).index(optimal_price)

    # Method used to sample a conversion rate from a customer class, given the price and the bid.
    # Returns the cr, the number of successes, the number of failures
    def sample_cr(self, class_n, price, bid):
        c = self.env.classes[class_n]
        new_clicks = np.ceil(self.env.distNewClicks.mean(c, bid))
        successes = self.env.distClickConverted.sample_n(c, price, new_clicks)
        cr = successes / new_clicks
        failures = new_clicks - successes

        return cr, successes, failures

    # Runs a round for a generic bandit. The learner must provide the price of the corresponding arm to be pulled,
    # this method returns the reward and the success/failures of advertising
    def round_bids_fixed(self, pulled_arm_val, bid, features=None):
        total_reward = 0
        successes_total = 0
        new_clicks_total = 0

        for c in self.env.classes:
            if not self.class_matches_filters(features, c):
                print("WTFFF")
                continue

            class_n = self.env.classes.index(c)

            # First extract the cr
            cr, successes, failures = self.sample_cr(class_n, pulled_arm_val, bid)

            # get the reward
            rwd = self.get_class_reward(class_n, pulled_arm_val, bid, cr, new_clicks_rounded=True)

            # sum to the total
            total_reward += rwd
            successes_total += successes
            new_clicks_total += (successes + failures)

        return total_reward, successes_total, new_clicks_total - successes_total

    # Method used to check whether a class respects the feature filters provided
    def class_matches_filters(self, features, c: CustomerClass):
        if features is None:
            return True
        for f in c.features:
            res = True
            for i in range(len(features)):
                if features[i] is not None and features[i] != f[i]:
                    res = False
            if res:
                return True
        return False

    # Given a class, price and bid, returns the revenue
    def get_class_reward(self, class_n, price, bid, cr=None, new_clicks_rounded=False):
        cust_class = self.env.classes[class_n]

        n = self.env.distNewClicks.mean(cust_class, bid)
        if new_clicks_rounded:
            n = np.ceil(n)

        if cr is None:
            cr = self.env.distClickConverted.mean(cust_class, price)

        future_visits = self.env.distFutureVisits.mean(cust_class)
        cost = self.env.distCostPerClick.mean(bid)

        rev = simple_class_profit(self.env.margin(price), n, cr, future_visits, cost)

        return np.around(rev, 2)

    # Returns the total reward obtained by all classes, given a price, a bid and the cr
    def get_total_reward(self, price, bid, cr):
        return np.sum([self.get_class_reward(n, price, bid, cr) for n in range(len(self.env.classes))])

    # Returns the list of rewards obtained by a clairvoyant in the scenario of price learning, with fixed bid
    def get_clairvoyant_rewards_price(self, n_rounds, prices, bid):
        clairvoyant = [0]
        for t in range(n_rounds):
            optimal_reward = self.get_optimal_reward(prices, bid)
            clairvoyant.append(optimal_reward)
        return clairvoyant

    # Returns a list containing all the cumulative rewards of the clairvoyant in the scenario of price learning,
    # given a bid. Used mainly to compute regret.
    def get_clairvoyant_cumulative_rewards_price(self, n_rounds, prices, bid):
        clairvoyant = self.get_clairvoyant_rewards_price(n_rounds, prices, bid)
        return np.cumsum(clairvoyant)

    # Given a cumulative reward list, returns a list containing the cumulative regret. It's for pricing problem
    def get_cumulative_regret_price(self, prices, bid, rewards):
        n_rounds = len(rewards) - 1
        return np.array(self.get_clairvoyant_cumulative_rewards_price(n_rounds, prices, bid) - np.array(rewards))
