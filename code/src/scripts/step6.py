import numpy as np

from src.Environment import Environment
from src.bandit.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.UCBOptimalJointLearner import UCBOptimalJointLearner


def main():
    prices = np.arange(10, 101, 10)
    bids = np.linspace(1, 100, num=10, dtype=np.int64)

    env = Environment(375411984)
    print(f'Running with seed {env.get_seed()}')
    n_rounds = 1000
    future_visits_delay = 30

    bandit_env = JointBanditEnvironment(env, prices, bids, future_visits_delay)
    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env.get_clairvoyant_cumulative_profits_not_discriminating(n_rounds)

    ucb_learner = UCBOptimalJointLearner(bandit_env, lambda r: r % 400 == 401)
    ucb_learner.learn(n_rounds)
    #ucb_cumulative_profits = ucb_learner.compute_cumulative_profits()


if __name__ == "__main__":
    main()