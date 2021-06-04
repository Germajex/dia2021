from src.constants import _Const
from src.CustomerClass import CustomerClass
import itertools
from numpy.random import Generator, default_rng
import numpy as np


class CustomerClassCreator:
    def get_new_classes(self, rng_generator: Generator, n_classes=None):
        CONST = _Const()
        customer_classes = []

        joint_features = list(itertools.product([True, False], repeat=2))
        rng_generator.shuffle(joint_features)

        n_classes = n_classes
        if n_classes is None:
            n_classes = CONST.N_CUSTOMER_CLASSES

        for i in range(n_classes):
            # Give joint features to the class
            max_features_pop = (len(joint_features) - (n_classes-i-1))
            n_joints = rng_generator.integers(1, max_features_pop + 1)
            features = []
            for j in range(n_joints):
                features.append(joint_features.pop())

            # Sample random parameters for each class
            new_clicks_r = rng_generator.integers(CONST.NEWCLICKS_MIN_R, CONST.NEWCLICKS_MAX_R)

            sigmoid_z = rng_generator.choice(CONST.SIGMOID_Z_VALUES_CR)
            cr_center = rng_generator.integers(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX)

            back_mean = rng_generator.integers(CONST.BACK_MEAN_MIN, CONST.BACK_MEAN_MAX)
            back_dev = rng_generator.integers(CONST.BACK_DEV_MIN, CONST.BACK_DEV_MAX)

            customer_classes.append(CustomerClass(CONST.NAMES[i], features, new_clicks_r, cr_center, sigmoid_z, back_mean, back_dev))

        return customer_classes

    def get_new_clicks_v_parameters(self, rng_generator: Generator):
        CONST = _Const()
        possible_bids_centers = np.linspace(CONST.BID_MIN, CONST.BID_MAX, 100)
        new_clicks_c = np.around(rng_generator.choice(possible_bids_centers), 2)
        new_clicks_z = rng_generator.choice(CONST.SIGMOID_Z_VALUES_NC)
        return new_clicks_c, new_clicks_z


if __name__ == '__main__':
    rng = default_rng(seed=1234)

    creator = CustomerClassCreator()
    classes = creator.get_new_classes(rng)

    for c in classes:
        c.print_summary()