import numpy as np


def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()

    return property(fget, fset)


class _Const(object):
    @constant
    def N_CUSTOMER_CLASSES():
        return 3

    @constant
    def NAMES():
        return ['C1', 'C2', 'C3']

    @constant
    def BID_MIN():
        return 1

    @constant
    def BID_MAX():
        return 20

    @constant
    def NEWCLICKS_MIN_R():
        return 5

    @constant
    def NEWCLICKS_MAX_R():
        return 50

    @constant
    def SIGMOID_Z_VALUES_NC():
        return [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    @constant
    def SIGMOID_Z_VALUES_CR():
        return [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]

    @constant
    def CR_CENTER_MIN():
        return 1

    @constant
    def CR_CENTER_MAX():
        return 100

    @constant
    def BACK_MEAN_MIN():
        return 0

    @constant
    def BACK_MEAN_MAX():
        return 10

    @constant
    def BACK_DEV_MIN():
        return 1

    @constant
    def BACK_DEV_MAX():
        return 5
