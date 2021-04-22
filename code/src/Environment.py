from numpy.random import default_rng, Generator


class Customer:
    def __init__(self, feature1: bool, feature2: bool):
        self.feature1 = feature1
        self.feature2 = feature2


class Environment:
    def __init__(self, random_seed=None):
        if random_seed is None:
            self.rng: Generator = default_rng()
        else:
            self.rng: Generator = default_rng(seed=random_seed)

        # TODO self.customerClasses = CustomerClasses(self.rng)
        # TODO spostare dentro a CustomerClasses?
