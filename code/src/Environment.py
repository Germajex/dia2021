import itertools
from numpy.random import default_rng


class Customer:
    def __init__(self, feature1: bool, feature2: bool):
        self.feature1 = feature1
        self.feature2 = feature2


class Environment:
    def __init__(self, random_seed=1234):
        self.rng = default_rng(seed=random_seed)

        # TODO self.customerClasses = CustomerClasses(self.random)
        # TODO spostare dentro a CustomerClasses?
        joint_features = list(itertools.product([True, False], repeat=2))
        self.rng.shuffle(joint_features)
        self.c1 = joint_features[:2]
        self.c2 = joint_features[2:3]
        self.c3 = joint_features[3:4]
