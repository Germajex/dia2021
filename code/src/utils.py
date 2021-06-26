import math


def sigmoid(x, center, z):
    return 1 / (1 + math.exp(-z * (x - center)))
