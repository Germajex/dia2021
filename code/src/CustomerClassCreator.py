from constants import _Const
from CustomerClass import CustomerClass
import itertools
import numpy as np
import random

class CustomerClassCreator:
    def __init__(self):
        random.seed()

    def getNewClasses(self):
        CONST = _Const()
        customerClasses = []

        joint_features = list(itertools.product([True, False], repeat=2))
        random.shuffle(joint_features)

        for i in range(CONST.N_CUSTOMER_CLASSES):
            # Give joint features to the class
            n_joints = random.randint(1, (len(joint_features) - (CONST.N_CUSTOMER_CLASSES-i-1)))
            features = []
            for j in range(n_joints):
                features.append(joint_features.pop())

            # Sample random parameters for each class
            newClicksM = random.randint(CONST.NEWCLICKS_MIN_M, CONST.NEWCLICKS_MAX_M)
            newClicksQ = random.randint(CONST.NEWCLICKS_MIN_Q, CONST.NEWCLICKS_MAX_Q)

            sigmoidZ = random.choice(CONST.SIGMOID_Z_VALUES)
            crCenter = random.randint(CONST.CR_CENTER_MIN, CONST.CR_CENTER_MAX)

            backMean = random.randint(CONST.BACK_MEAN_MIN, CONST.BACK_MEAN_MAX)
            backDev = random.randint(CONST.BACK_DEV_MIN, CONST.BACK_DEV_MAX)

            customerClasses.append(CustomerClass(CONST.NAMES[i], features, newClicksM, newClicksQ, crCenter, sigmoidZ, backMean, backDev))

        return customerClasses
