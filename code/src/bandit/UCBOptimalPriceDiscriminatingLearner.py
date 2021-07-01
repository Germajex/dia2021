from src.bandit.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.OptimalPriceDiscriminatingLearner import OptimalPriceDiscriminatingLearner
from src.bandit.UCBContext import UCBContext


class UCBOptimalPriceDiscriminatingLearner(OptimalPriceDiscriminatingLearner):
    def __init__(self, env: PriceBanditEnvironment):
        super().__init__(env, context_creator=lambda *args, **kwargs: UCBContext(*args, **kwargs))
