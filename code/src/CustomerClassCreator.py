from constants import _Const
from CustomerClass import CustomerClass
import itertools
from numpy.random import Generator, default_rng


class CustomerClassCreator:
    def getNewClasses(self, rng: Generator):
        CONST = _Const()
        customerClasses = []

        joint_features = list(itertools.product([True, False], repeat=2))
        rng.shuffle(joint_features)

        for i in range(CONST.N_CUSTOMER_CLASSES):
            # Give joint features to the class
            n_joints = rng.integers(1, (len(joint_features) - (CONST.N_CUSTOMER_CLASSES-i-1)))
            features = []
            for j in range(n_joints):
                features.append(joint_features.pop())

            # Sample random parameters for each class
            newClicksM = rng.integers(CONST.NEWCLICKS_MIN_M, CONST.NEWCLICKS_MAX_M)
            newClicksQ = rng.integers(CONST.NEWCLICKS_MIN_Q, CONST.NEWCLICKS_MAX_Q)

            sigmoidZ = rng.choice(CONST.SIGMOID_Z_VALUES)
            crCenter = rng.integers(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX)

            backMean = rng.integers(CONST.BACK_MEAN_MIN, CONST.BACK_MEAN_MAX)
            backDev = rng.integers(CONST.BACK_DEV_MIN, CONST.BACK_DEV_MAX)

            customerClasses.append(CustomerClass(CONST.NAMES[i], features, newClicksM, newClicksQ, crCenter, sigmoidZ, backMean, backDev))

        return customerClasses


if __name__ == '__main__':
    rng = default_rng(seed=1234)

    creator = CustomerClassCreator()
    classes = creator.getNewClasses(rng)

    for c in classes:
        c.printSummary()