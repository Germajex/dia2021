from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.learner.OptimalPriceDiscriminatingLearner import OptimalPriceDiscriminatingLearner
from src.bandit.context.TSContext import TSContext


# start ts disc
class TSOptimalPriceDiscriminatingLearner(OptimalPriceDiscriminatingLearner):
    def __init__(self, env: PriceBanditEnvironment):
        super().__init__(env, context_creator=lambda *args, **kwargs: TSContext(*args, **kwargs))
# end ts disc
