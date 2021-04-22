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
    def NEWCLICKS_MIN_M():
        return 1
    @constant
    def NEWCLICKS_MAX_M():
        return 5
    @constant
    def NEWCLICKS_MIN_Q():
        return 0
    @constant
    def NEWCLICKS_MAX_Q():
        return 5

    @constant
    def SIGMOID_Z_VALUES():
        return [0.3, 0.5, 0.8, 1, 1.5, 2]
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



