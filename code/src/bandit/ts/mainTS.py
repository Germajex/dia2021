from code.src.bandit.BanditEnvironment import BanditEnvironment
from TSLearner import TSLearner
from code.src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import code.src.bandit.LearningStats as stats

rng = np.random.default_rng()
creator = CustomerClassCreator()

N_ROUNDS = 365

customer = creator.getNewClasses(rng, 1)
customer[0].printSummary()

env = BanditEnvironment(10, customer)
learner = TSLearner(10, env)

learner.learn_price(N_ROUNDS, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 10)

stats.plot_results(["ts"], [learner.partial_rewards], learner.prices, learner.bids, N_ROUNDS, env)
