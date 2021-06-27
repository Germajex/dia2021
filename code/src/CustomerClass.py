import numpy.random.mtrand

from src.utils import sigmoid


class CustomerClass:
    def __init__(self, name, features, new_clicks_r, cr_center, sigmoid_z, back_dev,
                 back_mean):
        # Name
        self.name = name

        # Binary Features
        self.features = features

        # New user click
        self.newClicksR = new_clicks_r

        # Conversion rate (specular sigmoid function, centered in an arbitrary point
        self.crCenter = cr_center
        self.sigmoidZ = sigmoid_z

        # Number of times the user will come back (normal distribution)
        self.backMean = back_mean
        self.backDev = back_dev

    def print_summary(self):
        print(f"Class name: {self.name}")
        print('Binary features: ', *self.features)
        print(f"Daily auctions: {self.newClicksR}")
        print(f"Reserve price: {self.crCenter}, z={self.sigmoidZ}")
        print(f"Average future visits: {self.backMean}")

    def get_name(self):
        return self.name
