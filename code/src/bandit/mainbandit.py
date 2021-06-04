from BanditEnvironment import BanditEnvironment
from ts.TSLearner import TSLearner
from ucb.UCB1Learner import UCB1Learner
from src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import LearningStats as Stats

# seed=0xdeadbeef
rng = np.random.default_rng()
creator = CustomerClassCreator()

# Fixed variables for our problem
N_ROUNDS = 1000
arms = np.linspace(10, 100, 10)
n_arms = len(arms)
bid = 10

print("\n > > Starting simulation... < <\n")

# Create customer and environment
env = BanditEnvironment(n_arms, 3, rng)

if True:
    # Show summary of the customer classes
    for c in env.classes:
        c.print_summary()

optimal_arm = env.get_optimal_arm(arms, bid)
optimal_reward = np.mean(env.get_clairvoyant_rewards_price(N_ROUNDS, arms, bid))
optimal_cr = env.get_optimal_cr(arms, bid)
print(f"\nOptimal arm is n°{optimal_arm}")
print(f"With expected reward {optimal_reward:.2f}")
print(f"And expected cr {optimal_cr:.2f}")
clairvoyant = env.get_clairvoyant_cumulative_rewards_price(N_ROUNDS, arms, bid)

# Test algorithms
tsLearner_rwd = TSLearner(n_arms, 0.5, env)
tsLearner_cr = TSLearner(n_arms, 0, env)
ucbLearner_cr = UCB1Learner(n_arms, env)
ucbLearner_rwd = UCB1Learner(n_arms, env)

tsLearner_rwd.learn_price(N_ROUNDS, arms, bid, mode='rwd', verbose=True)
tsLearner_cr.learn_price(N_ROUNDS, arms, bid, mode='cr', verbose=True)
ucbLearner_cr.learn_price(N_ROUNDS, arms, bid, mode='cr', verbose=True)
ucbLearner_rwd.learn_price(N_ROUNDS, arms, bid, mode='rwd', verbose=True)

ts_rwd_reward = tsLearner_rwd.get_cumulative_rewards()
ts_cr_reward = tsLearner_cr.get_cumulative_rewards()
ucb_cr_reward = ucbLearner_cr.get_cumulative_rewards()
ucb_rwd_reward = ucbLearner_rwd.get_cumulative_rewards()

Stats.plot_results(["ts cr mode", "ts rwd mode", "ucb cr mode", "ucb rwd mode"],
                   [ts_cr_reward, ts_rwd_reward, ucb_cr_reward, ucb_rwd_reward], clairvoyant, N_ROUNDS, smooth=True)
