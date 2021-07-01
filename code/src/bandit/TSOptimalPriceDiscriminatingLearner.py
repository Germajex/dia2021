from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.OptimalPriceDiscriminatingLearner import OptimalPriceDiscriminatingLearner
from src.bandit.TSContext import TSContext
from src.bandit.UCBContext import UCBContext


class TSOptimalPriceDiscriminatingLearner(OptimalPriceDiscriminatingLearner):
    def __init__(self, env: PriceBanditEnvironment):
        super().__init__(env, context_creator=lambda *args, **kwargs: TSContext(*args, **kwargs))
