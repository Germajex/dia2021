from src.bandit.BanditEnvironment import BanditEnvironment
from TSLearner import TSLearner
import numpy as np
import src.bandit.LearningStats as Stats

rng = np.random.default_rng()

N_ROUNDS = 1000
N_RUNS = 10

rhos = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000]
prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_arms = len(prices)
bid = 10

learners = []
env = BanditEnvironment(10, 3, rng)
for rho in rhos:
    learners.append(TSLearner(n_arms, rho, env))

clairvoyant = env.get_clairvoyant_cumulative_rewards_price(N_ROUNDS, prices, bid)
rewards = []

for learner in learners:
    learner.learn_price(N_ROUNDS, prices, bid, "rwd")
    rewards.append(learner.get_cumulative_rewards())

Stats.plot_results(rhos, rewards, clairvoyant, N_ROUNDS)



