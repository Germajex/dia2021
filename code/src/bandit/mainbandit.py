from code.src.bandit.BanditEnvironment import BanditEnvironment
from ts.TSLearner import TSLearner
from ucb.UCB1Learner import UCB1Learner
from code.src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import code.src.bandit.LearningStats as Stats

rng = np.random.default_rng()
creator = CustomerClassCreator()

# Fixed variables for our problem
N_ROUNDS = 365
N_RUNS = 10
arms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bid = 7

# Rewards
ts_rewards = []
ucb_cr_rewards = []
ucb_rwd_rewards = []

clairvoyants = []

print("\n > > Starting simulation... < <\n")

# Runs the learning algorithm N_RUNS times in order to average the results
for r in range(N_RUNS):

    print(f"\n[info] Run n°{r+1} ...")
    # Create customer and environment
    env = BanditEnvironment(10, 3, rng)

    if False:
        # Show summary of the customer classes
        for c in env.classes:
            c.printSummary()

    clairvoyants.append(env.get_clairvoyant_partial_rewards_price(N_ROUNDS, bid))

    # Test algorithms
    tsLearner = TSLearner(10, 0.05, env)
    ucbLearner_cr = UCB1Learner(10, env)
    ucbLearner_rwd = UCB1Learner(10, env)

    tsLearner.learn_price(N_ROUNDS, arms, bid, verbose=False)
    ucbLearner_cr.learn_price(N_ROUNDS, arms, bid, mode='cr', verbose=False)
    ucbLearner_rwd.learn_price(N_ROUNDS, arms, bid, mode='rwd', verbose=False)

    ts_rewards.append(tsLearner.partial_rewards)
    ucb_cr_rewards.append(ucbLearner_cr.cumulative_rewards)
    ucb_rwd_rewards.append(ucbLearner_rwd.cumulative_rewards)

# Average rewards and then plot
ts_rewards = np.mean(ts_rewards, axis=0)
ucb_cr_rewards = np.mean(ucb_cr_rewards, axis=0)
ucb_rwd_rewards = np.mean(ucb_rwd_rewards, axis=0)

clairvoyants = np.mean(clairvoyants, axis=0)

Stats.plot_results(["ts", "ucb cr mode", "ucb rwd mode"], [ts_rewards, ucb_cr_rewards, ucb_rwd_rewards],
                   clairvoyants, N_ROUNDS)
