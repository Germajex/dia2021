from numpy.random import default_rng, Generator
from src.constants import _Const
from src.CustomerClassCreator import CustomerClassCreator
from src.distributions import NewClicksDistribution, ClickConvertedDistribution, FutureVisitsDistribution, \
    CostPerClickDistribution


class Environment:
    def __init__(self, random_seed=None):
        CONST = _Const()

        if random_seed is None:
            self._rng: Generator = default_rng()
            random_seed = self._rng.integers(0, 2 ** 32)

        self._seed = random_seed
        self._rng: Generator = default_rng(seed=random_seed)

        self.classes = CustomerClassCreator().get_new_classes(self._rng, CONST.N_CUSTOMER_CLASSES)

        self.newClicksC, self.newClicksZ = CustomerClassCreator().get_new_clicks_v_parameters(self._rng)

        self.distNewClicks = NewClicksDistribution(self._rng, self.newClicksC, self.newClicksZ)
        self.distClickConverted = ClickConvertedDistribution(self._rng)
        self.distFutureVisits = FutureVisitsDistribution(self._rng)
        self.distCostPerClick = CostPerClickDistribution(self._rng)

    def get_seed(self):
        return self._seed

    @staticmethod
    def margin(price: float):
        return price
