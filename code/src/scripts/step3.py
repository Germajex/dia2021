from src.Environment import Environment
from src.bandit.BanditEnvironment import BanditEnvironment
from src.bandit.ucb.UCB1Learner import UCB1Learner


def main():
    n_prices = 10
    env = Environment()
    bandit_env = BanditEnvironment(environment=env, n_arms=10)

    learner = UCB1Learner(10, bandit_env)














if __name__ == "__main__":
    main()