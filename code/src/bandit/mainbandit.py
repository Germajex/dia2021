from BanditEnvironment import BanditEnvironment
from src.Environment import Environment
from ts.TSLearner import TSLearner
from ucb.UCBLearner import UCBLearner
from src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import LearningStats as Stats

# Fixed variables for our problem
N_ROUNDS = 365
prices = np.linspace(10, 100, 10)
n_arms = len(prices)
bid = 10

print("\n > > Starting simulation... < <\n")

# Create customer and environment
env = Environment(random_seed=None)
print(f'Generated environment with seed {env.get_seed()}')
bandit_env = BanditEnvironment(environment=env, n_arms=n_arms)

if True:
    env.print_class_summary()

optimal_arm = bandit_env.get_optimal_arm(prices, bid)
optimal_reward = np.mean(bandit_env.get_clairvoyant_rewards_price(N_ROUNDS, prices, bid))
optimal_cr = bandit_env.get_optimal_cr(prices, bid)
print(f"\nOptimal arm is n°{optimal_arm}")
print(f"With expected reward {optimal_reward:.2f}")
print(f"And expected cr {optimal_cr:.2f}")
clairvoyant = bandit_env.get_clairvoyant_cumulative_rewards_price(N_ROUNDS, prices, bid)

# Test algorithms
tsLearner_rwd = TSLearner(n_arms, 1, bandit_env)
tsLearner_cr = TSLearner(n_arms, 0, bandit_env)
ucbLearner_cr = UCBLearner(n_arms, bandit_env)
ucbLearner_rwd = UCBLearner(n_arms, bandit_env)

tsLearner_rwd.learn_price(N_ROUNDS, prices, bid, mode='rwd', verbose=True)
tsLearner_cr.learn_price(N_ROUNDS, prices, bid, mode='cr', verbose=True)
ucbLearner_cr.learn_price(N_ROUNDS, prices, bid, mode='cr', verbose=True)
ucbLearner_rwd.learn_price(N_ROUNDS, prices, bid, mode='rwd', verbose=True)

ts_rwd_reward = tsLearner_rwd.get_cumulative_rewards()
ts_cr_reward = tsLearner_cr.get_cumulative_rewards()
ucb_cr_reward = ucbLearner_cr.get_cumulative_rewards()
ucb_rwd_reward = ucbLearner_rwd.get_cumulative_rewards()

Stats.plot_results(["ts cr mode", "ts rwd mode", "ucb cr mode", "ucb rwd mode"],
                   [ts_cr_reward, ts_rwd_reward, ucb_cr_reward, ucb_rwd_reward], clairvoyant, N_ROUNDS, smooth=True)
