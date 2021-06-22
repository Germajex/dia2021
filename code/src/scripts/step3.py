from src.Environment import Environment
from src.algorithms import step1
from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.ucb.UCBLearner import UCBLearner
import numpy as np

def main():
    prices = np.arange(10, 101, 10)
    bids = np.arange(1, 100)

    env = Environment()

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = BanditEnvironment(env, prices, opt_bid, 30)

    learner = UCBLearner(bandit_env)

    learner.learn2(100)

    print(learner.compute_projected_profits())













if __name__ == "__main__":
    main()