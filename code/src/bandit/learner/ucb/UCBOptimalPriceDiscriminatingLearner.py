from src.bandit.banditEnvironments.PriceBanditEnvironment import PriceBanditEnvironment
from src.bandit.learner.OptimalPriceDiscriminatingLearner import OptimalPriceDiscriminatingLearner
from src.bandit.context.UCBContext import UCBContext


# start ucbdisc
class UCBOptimalPriceDiscriminatingLearner(OptimalPriceDiscriminatingLearner):
    def __init__(self, env: PriceBanditEnvironment):
        super().__init__(env,
                         context_creator=lambda *args,
                                                **kwargs:
                         UCBContext(*args, **kwargs))

# end ucbdisc
