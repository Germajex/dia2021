from code.src.bandit.BanditEnvironment import BanditEnvironment
from ts.TSLearner import TSLearner
from ucb.UCB1Learner import UCB1Learner
from code.src.CustomerClassCreator import CustomerClassCreator
import numpy as np
import code.src.bandit.LearningStats as Stats

rng = np.random.default_rng()
creator = CustomerClassCreator()

# Fixed variables for our problem
N_ROUNDS = 1000
arms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bid = 15

customer = creator.getNewClasses(rng, 1)
customer[0].printSummary()

env = BanditEnvironment(10, customer)

tsLearner = TSLearner(10, env)
ucbLearner = UCB1Learner(10, env)

print("\n > > Starting simulation... < <\n")

tsLearner.learn_price(N_ROUNDS, arms, bid)
ucbLearner.learn_price(N_ROUNDS, arms, bid)

Stats.plot_results(["ts", "ucb"], [tsLearner.partial_rewards, ucbLearner.partial_rewards], arms, [bid], N_ROUNDS, env)
