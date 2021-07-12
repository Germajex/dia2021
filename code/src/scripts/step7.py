import numpy as np

from src.Environment import Environment
from src.algorithms import step1
from src.bandit.LearningStats import plot_results
from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.context.UCBJointContext import UCBJointContext
from src.bandit.learner.OptimalJointDiscriminatingLearner import OptimalJointDiscriminatingLearner
from src.bandit.learner.ucb.UCBOptimalPriceDiscriminatingLearner import UCBOptimalPriceDiscriminatingLearner


def main():
    prices = np.linspace(10, 100, num=10, dtype=np.int64)
    bids = np.linspace(1, 40, num=10, dtype=np.int64)
    delay = 30
    step_4_n_rounds = 400
    n_rounds = 400

    env_for_step4 = Environment(3442565254)
    seed = env_for_step4.get_seed()
    env = Environment(seed)

    _, opt_bid, _ = step1(env_for_step4, prices, bids)
    bandit_env_step4 = PriceBanditEnvironment(env_for_step4, prices, opt_bid, delay)

    print(f'Running step4 (with seed {seed}) to find context structure')
    ucb_disc_learner = UCBOptimalPriceDiscriminatingLearner(bandit_env_step4)
    ucb_disc_learner.learn(step_4_n_rounds)
    contexts = ucb_disc_learner.context_structure

    print(f'Context structure found!')
    feature_combs = [comb.features for comb in contexts]
    for i in range(len(feature_combs)):
        print(f'Context {i}: {feature_combs[i]}')

    context_structure = [UCBJointContext(comb, env_for_step4.margin, len(prices), len(bids), env_for_step4.rng) for comb in feature_combs]

    print(f'Running step7 (with seed {seed}) to find optimal joint bidding/pricing strategy')
    bandit_env = JointBanditEnvironment(env, prices, bids, delay)
    joint_disc_learner = OptimalJointDiscriminatingLearner(bandit_env, context_structure)
    joint_disc_learner.learn(n_rounds)

    regret_length = n_rounds - delay
    clairvoyant = bandit_env.get_clairvoyant_cumulative_profit_discriminating(n_rounds)
    strategies = joint_disc_learner.get_strategies()
    learner_profit = bandit_env.get_learner_cumulative_profit_discriminating(strategies)

    plot_results(["ucb"], [learner_profit], clairvoyant, n_rounds, smooth=False)


if __name__ == "__main__":
    main()
