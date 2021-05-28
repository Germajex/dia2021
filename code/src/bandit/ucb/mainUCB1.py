from code.src.bandit.BanditEnvironment import BanditEnvironment
from UCB1Learner import UCB1Learner
from code.src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import code.src.bandit.LearningStats as Stats

N_RUNS = 10
N_ROUNDS = 365

rng = np.random.default_rng()
creator = CustomerClassCreator()

prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bid = 10
# cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
cs = [10, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000]

rewards = [[] for c in cs]
clairvoyants = []

# Run the learning many times, to avg results
for i in range(N_RUNS):
    print(f'[info] Running run n°{i+1}')

    # Create environment and compute clairvoyant results
    env = BanditEnvironment(10, 3, rng)
    clairvoyants.append(env.get_clairvoyant_partial_rewards_price(N_ROUNDS, bid))
    learners = []

    # Create the learners
    for j in range(len(cs)):
        learners.append(UCB1Learner(10, env))

    # Run all the learners and save their results
    for learner in learners:
        learner.learn_price(N_ROUNDS, prices, bid, mode='rwd', c_param=cs[learners.index(learner)], verbose=False)
        rewards[learners.index(learner)].append(learner.cumulative_rewards)

# Numpy stuff to average the various runs of the learners
for k in range(len(rewards)):
    rewards[k] = np.mean(rewards[k], axis=0)
# Also average the clairvoyant results
clairvoyants = np.mean(clairvoyants, axis=0)

# Finally plot results
Stats.plot_results(cs, rewards, clairvoyants, N_ROUNDS)
