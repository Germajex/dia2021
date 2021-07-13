import numpy as np
from terminaltables import AsciiTable

from src.Environment import Environment
from src.algorithms import step1
from src.bandit.LearningStats import plot_results
from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment
from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.context.UCBJointContext import UCBJointContext
from src.bandit.learner.OptimalJointDiscriminatingLearner import OptimalJointDiscriminatingLearner
from src.bandit.learner.ucb.UCBOptimalPriceDiscriminatingLearner import UCBOptimalPriceDiscriminatingLearner
from src.scripts.environment_plotter import plot_everything


def main():

    prices = np.linspace(10, 100, num=10, dtype=np.int64)
    bids = np.linspace(1, 60, num=10, dtype=np.int64)
    delay = 30
    step_4_n_rounds = 400
    n_rounds = 365
    interactive = False

    for envN, seedV in enumerate([4264432570, 1530294961, 1966222620]):
        env = Environment(random_seed=seedV) if seedV is not None else Environment()
        print(f'Running with seed {env.get_seed()}')
        env_for_step4 = Environment(seedV)

        _, opt_bid, _ = step1(env_for_step4, prices, bids)
        bandit_env_step4 = PriceBanditEnvironment(env_for_step4, prices, opt_bid, delay)

        print(f'Running step4 (with seed {seedV}) to find context structure')
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
        context_structure = [
            UCBJointContext(comb, bandit_env.margin, len(prices), len(bids), bandit_env.rng)
            for comb in feature_combs
        ]

        print(f'Running step7 (with seed {seedV}) to find optimal joint bidding/pricing strategy')

        joint_disc_learner = OptimalJointDiscriminatingLearner(bandit_env, context_structure)
        joint_disc_learner.learn(n_rounds)

        clairvoyant = bandit_env.get_clairvoyant_cumulative_profit_discriminating(n_rounds)
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

        env.print_summary()

        if interactive:
            plot_results(["UCB"], [learner_profit], clairvoyant, n_rounds)

        pull_tables = []
        for context in joint_disc_learner.context_structure:
            print('Context:',  *context.features)
            pulls = [[0 for b in bids] for p in prices]
            comb = context.features[0]
            for p_a, b_a in joint_disc_learner.get_strategies():
                pulls[p_a[comb]][b_a[comb]] += 1

            table_pull_data = [['P\\B'] + [f'{b:.2f}' for b in bids]]
            for p in prices:
                table_pull_data.append([f'{p:.2f}'])
            for p_i, p in enumerate(prices):
                for b_i, b in enumerate(bids):
                    table_pull_data[p_i+1].append(f'{pulls[p_i][b_i]}')

            table_pull = AsciiTable(table_pull_data)
            for i in range(len(table_pull_data[0])):
                table_pull.justify_columns[i] = 'right'

            pull_tables.append(table_pull)
            print(table_pull.table)

            expected_profs = np.array([[0 for b in bids] for p in prices])
            for comb in context.features:
                pass


        dir = '../../../report/figures/step7'

        with open(dir+f'/output{envN}.txt', 'w', encoding='utf8') as output_file:
            output_file.write(f' Seed: {seedV}\n')
            output_file.write(f' UCB number of pulls:\n')

        plot_results(["UCB"], [learner_profit], clairvoyant, n_rounds,
                     dest_file_path=dir+f'/plot{envN}')

        plot_everything(environment=env, path=dir+f'/env{envN}')



if __name__ == "__main__":
    main()
