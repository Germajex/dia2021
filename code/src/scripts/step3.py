from src.Environment import Environment
from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.ucb.UCBLearner import UCBLearner


def main():
    n_prices = 10
    env = Environment()
    bandit_env = BanditEnvironment(environment=env, n_arms=10)

    learner = UCBLearner(10, bandit_env)














if __name__ == "__main__":
    main()