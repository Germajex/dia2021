from src.bandit.BanditEnvironment import BanditEnvironment
from UCBLearner import UCBLearner
import numpy as np
import src.bandit.LearningStats as Stats
from src.Environment import Environment

N_RUNS = 5
N_ROUNDS = 700

rng = np.random.default_rng()

env = Environment()
print(f'Created environment with seed {env.get_seed()}')

prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bid = 10
cs = [0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1]
# cs = [10, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000]

rewards = [[] for c in cs]
clairvoyants = []

# Run the learning many times, to avg results
for i in range(N_RUNS):
    print(f'[info] Running run n°{i+1}')

    # Create environment and compute clairvoyant results
    bandit_env = BanditEnvironment(environment=env, n_arms=10)
    clairvoyants.append(bandit_env.get_clairvoyant_cumulative_rewards_price(N_ROUNDS, prices, bid))


    learners = []

    # Create the learners
    for j in range(len(cs)):
        learners.append(UCBLearner(10, bandit_env))

    # Run all the learners and save their results
    for learner in learners:
        learner.learn_price(N_ROUNDS, prices, bid, mode='rwd', c_param=cs[learners.index(learner)], verbose=False)
        rewards[learners.index(learner)].append(learner.get_cumulative_rewards())

# Numpy stuff to average the various runs of the learners
for k in range(len(rewards)):
    rewards[k] = np.mean(rewards[k], axis=0)
# Also average the clairvoyant results
clairvoyants = np.mean(clairvoyants, axis=0)

# Finally plot results
Stats.plot_results(cs, rewards, clairvoyants, N_ROUNDS)
