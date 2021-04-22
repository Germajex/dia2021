import numpy.random.mtrand

from utils import sigmoid

class CustomerClass:
    def __init__(self, name, features, newClicksM, newClicksQ, crCenter, sigmoidZ, backDev, backMean):
        # Name
        self.name = name

        # Binary Features
        self.features = features

        # New user click (linear function)
        self.newClicks = [newClicksM, newClicksQ]
        # Conversion rate (specular sigmoid function, centered in an arbitrary point
        self.crCenter = crCenter
        self.sigmoidZ = sigmoidZ
        # Number of times the user will come back (normal distribution)
        self.backMean = backMean
        self.backDev = backDev

    def getNewClicks(self, bid):
        return bid*self.newClicks[0] + self.newClicks[1]

    def getConversionRate(self, price):
        return sigmoid(-price, self.crCenter, self.sigmoidZ)

    def getCustomerBackProb(self):
        return numpy.random.mtrand.normal(self.backMean, self.backDev)

    def printSummary(self):
        print("--- ooo ---")
        print("Class name: {}".format(self.name))
        print("Binary features: {}".format(self.features))
        print("New clicks params: m={}, q={}".format(self.newClicks[0], self.newClicks[1]))
        print("Conversion rate params: center={}, z={}".format(self.crCenter, self.sigmoidZ))
        print("Client fidelity params: mean={}, variance={}".format(self.backMean, self.backDev))