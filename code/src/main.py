from CustomerClassCreator import CustomerClassCreator
from numpy.random import default_rng

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rng = default_rng(seed=1234)

    creator = CustomerClassCreator()
    classes = creator.getNewClasses()

    for c in classes:
        c.printSummary()
