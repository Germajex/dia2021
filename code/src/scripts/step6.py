import numpy as np

from src.Environment import Environment
from src.algorithms import step1
from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.LearningStats import plot_results
from src.bandit.learner.ucb.UCBOptimalJointLearner import UCBOptimalJointLearner


def main():
    # [!] Feel free to play with the number of arms [!]
    prices = np.linspace(10, 100, num=15, dtype=np.int64)
    bids = np.linspace(1, 40, num=10, dtype=np.int64)

    env = Environment()
    print(f'Running with seed {env.get_seed()}')
    n_rounds = 1500
    future_visits_delay = 30

    opt_price, opt_bid, profit = step1(env, prices, bids)

    bandit_env = JointBanditEnvironment(env, prices, bids, future_visits_delay)
    regret_length = n_rounds - future_visits_delay
    clairvoyant_cumulative_profits = bandit_env.get_clairvoyant_cumulative_profits_not_discriminating(regret_length)

    ucb_learner = UCBOptimalJointLearner(bandit_env, lambda r: r % 400 == 401)
    ucb_learner.learn(n_rounds)
    ucb_cumulative_profits = bandit_env.get_learner_cumulative_profit_not_discriminating(ucb_learner.pulled_arms)

    print(f'Optimal price is {opt_price} and optimal bid is {opt_bid}')

    ucb_recap = ucb_learner.get_pulled_arms_recap()
    print(f'Bids ->  | ' + ' '.join(f'{bandit_env.bids[b]:5d}' for b in range(ucb_learner.n_arms_bid)))
    print(f'Prices v | ' + '-'*6*ucb_learner.n_arms_bid)
    for arm_p in range(ucb_learner.n_arms_price):
        print(f'{bandit_env.prices[arm_p]:3d}      | ' + ' '.join(f'{p:5d}' for p in ucb_recap[arm_p]))

    plot_results(["ucb"], [ucb_cumulative_profits], clairvoyant_cumulative_profits, regret_length, smooth=False)


if __name__ == "__main__":
    main()
