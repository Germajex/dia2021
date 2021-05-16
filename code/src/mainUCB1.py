from BanditEnviroment import BanditEnviroment
from UCB1Learner import UCB1Learner
from CustomerClassCreator import CustomerClassCreator
import numpy as np

rng = np.random.default_rng()
creator = CustomerClassCreator()

customer = creator.getNewClasses(rng, 1)
customer[0].printSummary()

env = BanditEnviroment(5, customer)
learner = UCB1Learner(5, env)

learner.learn_price(1000, [10, 20, 50, 70, 90], 10)