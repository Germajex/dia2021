import numpy.random.mtrand
import numpy as np

from code.src.utils import sigmoid

class CustomerClass:
    def __init__(self, name, features, newClicksR, newClicksCenter, newClicksZ, crCenter, sigmoidZ, backDev, backMean):
        # Name
        self.name = name

        # Binary Features
        self.features = features

        # New user click (linear function)
        self.newClicksR = newClicksR
        self.newClicksC = newClicksCenter
        self.newClicksZ = newClicksZ
        # Conversion rate (specular sigmoid function, centered in an arbitrary point
        self.crCenter = crCenter
        self.sigmoidZ = sigmoidZ
        # Number of times the user will come back (normal distribution)
        self.backMean = backMean
        self.backDev = backDev

    def getNewClicks(self, bid):
        return self.newClicksR*sigmoid(bid, self.newClicksC, self.newClicksZ)

    def getConversionRate(self, price):
        return sigmoid(-price, -self.crCenter, self.sigmoidZ)

    def getCustomerBackProb(self):
        return numpy.random.mtrand.normal(self.backMean, self.backDev)

    def getCustomerBackMean(self):
        return self.backMean

    def getRevenue(self, price, bid, cr=None):
        n = self.getNewClicks(bid)
        if cr is None:
            cr = self.getConversionRate(price)
        back = self.getCustomerBackMean()

        rev = (-bid * n) + (n * cr * price) + (n * back * cr * price)

        return np.around(rev, 2)

    def printSummary(self):
        print("--- ooo ---")
        print("Class name: {}".format(self.name))
        print("Binary features: {}".format(self.features))
        print("New clicks params: range={}, c={}, z={}".format(self.newClicksR, self.newClicksC, self.newClicksZ))
        print("Conversion rate params: center={}, z={}".format(self.crCenter, self.sigmoidZ))
        print("Client fidelity params: mean={}, variance={}".format(self.backMean, self.backDev))