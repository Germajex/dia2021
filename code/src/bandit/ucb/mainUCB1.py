from code.src.bandit.BanditEnvironment import BanditEnvironment
from UCB1Learner import UCB1Learner
from code.src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import code.src.bandit.LearningStats as stats

N_ROUNDS = 365

rng = np.random.default_rng()
creator = CustomerClassCreator()

customer = creator.getNewClasses(rng, 1)
customer[0].printSummary()

env = BanditEnvironment(10, customer)
learner = UCB1Learner(10, env)

learner.learn_price(N_ROUNDS, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 10)

stats.plot_results(["ucb1"], [learner.partial_rewards], learner.prices, learner.bids, N_ROUNDS, env)
