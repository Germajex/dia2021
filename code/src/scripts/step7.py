import numpy as np

from src.Environment import Environment
from src.algorithms import step1
from src.bandit.LearningStats import plot_results
from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.context.UCBJointContext import UCBJointContext
from src.bandit.learner.OptimalJointDiscriminatingLearner import OptimalJointDiscriminatingLearner
from src.bandit.learner.ucb.UCBOptimalPriceDiscriminatingLearner import UCBOptimalPriceDiscriminatingLearner
import warnings


def main():
    warnings.filterwarnings('error')
    prices = np.linspace(10, 100, num=10, dtype=np.int64)
    bids = np.linspace(1, 60, num=10, dtype=np.int64)
    delay = 30
    step_4_n_rounds = 365
    n_rounds = 365+300

    env_for_step4 = Environment()
    seed = env_for_step4.get_seed()
    env = Environment(seed)

    _, opt_bid, _ = step1(env_for_step4, prices, bids)
    bandit_env_step4 = PriceBanditEnvironment(env_for_step4, prices, opt_bid, delay)

    print(f'Running step4 (with seed {seed}) to find context structure')
    ucb_disc_learner = UCBOptimalPriceDiscriminatingLearner(bandit_env_step4)
    ucb_disc_learner.learn(step_4_n_rounds)
    contexts = ucb_disc_learner.context_structure

    print(f'Context structure found!')
    feature_combs = [[] for i in range(len(contexts))]
    for i in range(len(contexts)):
        for j in range(len(contexts[i].features)):
            feature_combs[i].append(contexts[i].features[j])
    for i in range(len(feature_combs)):
        print(f'Context {i}: {feature_combs[i]}')

    bandit_env = JointBanditEnvironment(env, prices, bids, delay)
    context_structure = [UCBJointContext(comb, bandit_env.margin, len(prices), len(bids), bandit_env.rng) for comb in
                         feature_combs]

    print(f'Running step7 (with seed {seed}) to find optimal joint bidding/pricing strategy')

    joint_disc_learner = OptimalJointDiscriminatingLearner(bandit_env, context_structure)
    joint_disc_learner.learn(n_rounds)

    clairvoyant = bandit_env.get_clairvoyant_cumulative_profit_discriminating(context_structure, n_rounds)
    strategies = joint_disc_learner.get_strategies()
    learner_profit = bandit_env.get_learner_cumulative_profit_discriminating(strategies)

    for i, context in enumerate(joint_disc_learner.get_context_structure()):
        recap = context.get_pulled_arms_recap()
        print(f'Context n°{i} - Combinations of features:', *context.features)
        print(f'Bids ->  | ' + ' '.join(f'{bandit_env.bids[b]:5d}' for b in range(context.n_arms_bid)))
        print(f'Prices v | ' + '-' * 6 * context.n_arms_bid)

        for arm_p in range(context.n_arms_price):
            print(f'{bandit_env.prices[arm_p]:3d}      | ' + ' '.join(f'{p:5d}' for p in recap[arm_p]))
        print()

    print(f'Expected profit: {learner_profit[-1]}')
    print()
    env.print_class_summary()

    plot_results(["ucb"], [learner_profit], clairvoyant, n_rounds, smooth=False)


if __name__ == "__main__":
    main()
