import numpy as np

from src.Environment import Environment
from src.bandit.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.LearningStats import plot_results
from src.bandit.UCBOptimalJointLearner import UCBOptimalJointLearner


def main():
    prices = np.arange(10, 101, 10)
    bids = np.linspace(1, 100, num=10, dtype=np.int64)

    env = Environment(3374247623)
    print(f'Running with seed {env.get_seed()}')
    n_rounds = 1400
    future_visits_delay = 30

    bandit_env = JointBanditEnvironment(env, prices, bids, future_visits_delay)
    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env.get_clairvoyant_cumulative_profits_not_discriminating(regret_length)

    ucb_learner = UCBOptimalJointLearner(bandit_env, lambda r: r % 400 == 401)
    ucb_learner.learn(n_rounds)
    ucb_cumulative_profits = bandit_env.get_learner_cumulative_profit_not_discriminating(ucb_learner.pulled_arms)

    ucb_recap = ucb_learner.get_pulled_arms_recap()
    print(f'Bids ->  | ' + ' '.join(f'{bandit_env.bids[b]:5d}' for b in range(ucb_learner.n_arms_bid)))
    print(f'Prices v | ' + '-'*6*ucb_learner.n_arms_bid)
    for arm_p in range(ucb_learner.n_arms_price):
        print(f'{bandit_env.prices[arm_p]:3d}      | ' + ' '.join(f'{p:5d}' for p in ucb_recap[arm_p]))

    plot_results(["UCB"], [ucb_cumulative_profits], clairvoyant_cumulative_profits, regret_length, smooth=True)


if __name__ == "__main__":
    main()
