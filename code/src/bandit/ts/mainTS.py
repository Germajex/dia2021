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

rhos = [0.01, 0.1, 0.5, 1, 10]
prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bid = 10

learners = []
env = BanditEnvironment(10, customer)
for rho in rhos:
    learners.append(TSLearner(10, rho, env))

rewards = []

for learner in learners:
    learner.learn_price(N_ROUNDS, prices, bid)
    rewards.append(learner.partial_rewards)

