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
N_RUNS = 2
arms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bid = 7

# Rewards
ts_rewards = []
ucb_rewards = []

ts_regrets = []
ucb_regrets = []

clairvoyants = []

print("\n > > Starting simulation... < <\n")

# Runs the learning algorithm N_RUNS times in order to average the results
for r in range(N_RUNS):

    print(f"\n[info] Run n°{r} ...")
    # Create customer and environment
    customers = creator.getNewClasses(rng, 3)
    env = BanditEnvironment(10, customers, bids=[bid])

    # Show summary of the customer classes
    for c in customers:
        c.printSummary()

    clairvoyants.append(env.get_clairvoyant_partial_rewards(N_ROUNDS))

    # Test algorithms
    tsLearner = TSLearner(10, 0.05, env)
    ucbLearner = UCB1Learner(10, env)

    tsLearner.learn_price(N_ROUNDS, arms, bid, verbose=True)
    ucbLearner.learn_price(N_ROUNDS, arms, bid, verbose=True)

    ts_rewards.append(tsLearner.partial_rewards)
    ucb_rewards.append(ucbLearner.partial_rewards)

    ts_regrets.append(env.get_cumulative_regret(tsLearner.partial_rewards))
    ucb_regrets.append(env.get_cumulative_regret(ucbLearner.partial_rewards))

# Average rewards and then plot
ts_rewards = np.mean(ts_rewards, axis=0)
ucb_rewards = np.mean(ucb_rewards, axis=0)
ts_regrets = np.mean(ts_regrets, axis=0)
ucb_regrets = np.mean(ucb_regrets, axis=0)

clairvoyants = np.mean(clairvoyants, axis=0)

Stats.plot_results(["ts", "ucb"], [ts_rewards, ucb_rewards], [ts_regrets, ucb_regrets], clairvoyants, N_ROUNDS)
