from numpy.random import default_rng, Generator
from src.constants import _Const
from src.CustomerClassCreator import CustomerClassCreator
from src.distributions import NewClicksDistribution, ClickConvertedDistribution, FutureVisitsDistribution, CostPerClickDistribution


class Environment:
    def __init__(self, random_seed=None):
        CONST = _Const()
        if random_seed is None:
            self.rng: Generator = default_rng()
            random_seed = self.rng.integers(0, 2**32)

        self.rng: Generator = default_rng(seed=random_seed)
        print(f'Created environment with seed {random_seed}')

        self.classes = CustomerClassCreator().getNewClasses(self.rng, CONST.N_CUSTOMER_CLASSES)

        self.distNewClicks = NewClicksDistribution(self.rng)
        self.distClickConverted = ClickConvertedDistribution(self.rng)
        self.distFutureVisits = FutureVisitsDistribution(self.rng)
        self.distCostPerClick = CostPerClickDistribution(self.rng)
