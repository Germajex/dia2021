import numpy as np

from src.bandit.OptimalPriceDiscriminatingLearner import OptimalPriceDiscriminatingLearner
from src.bandit.BanditEnvironment import BanditEnvironment

from src.bandit.UCBContext import UCBContext


class UCBOptimalPriceDiscriminatingLearner(OptimalPriceDiscriminatingLearner):
    def __init__(self, env: BanditEnvironment):
        super().__init__(env, context_creator=lambda *args, **kwargs: UCBContext(*args, **kwargs))
