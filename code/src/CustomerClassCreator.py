from code.src.constants import _Const
from code.src.CustomerClass import CustomerClass
import itertools
from numpy.random import Generator, default_rng
import numpy as np


class CustomerClassCreator:
    def getNewClasses(self, rng: Generator, n_classes=None):
        CONST = _Const()
        customerClasses = []

        joint_features = list(itertools.product([True, False], repeat=2))
        rng.shuffle(joint_features)

        # Parameters for new clicks sigmoid (same for each class)
        possible_bids_centers = np.linspace(CONST.BID_MIN, CONST.BID_MAX, 100)
        newClicksC = np.around(rng.choice(possible_bids_centers), 2)
        newClicksZ = rng.choice(CONST.SIGMOID_Z_VALUES_NC)

        n_classes = n_classes
        if n_classes is None:
            n_classes = CONST.N_CUSTOMER_CLASSES

        for i in range(n_classes):
            # Give joint features to the class
            max_features_pop = (len(joint_features) - (n_classes-i-1))
            n_joints = rng.integers(1, max_features_pop+1)
            features = []
            for j in range(n_joints):
                features.append(joint_features.pop())

            # Sample random parameters for each class
            newClicksR = rng.integers(CONST.NEWCLICKS_MIN_R, CONST.NEWCLICKS_MAX_R)

            sigmoidZ = rng.choice(CONST.SIGMOID_Z_VALUES_CR)
            crCenter = rng.integers(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX)

            backMean = rng.integers(CONST.BACK_MEAN_MIN, CONST.BACK_MEAN_MAX)
            backDev = rng.integers(CONST.BACK_DEV_MIN, CONST.BACK_DEV_MAX)

            customerClasses.append(CustomerClass(CONST.NAMES[i], features, newClicksR, newClicksC, newClicksZ, crCenter, sigmoidZ, backMean, backDev))

        return customerClasses


if __name__ == '__main__':
    rng = default_rng(seed=1234)

    creator = CustomerClassCreator()
    classes = creator.getNewClasses(rng)

    for c in classes:
        c.printSummary()