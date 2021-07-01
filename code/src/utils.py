import math

import numpy as np


def sigmoid(x, center, z):
    return 1 / (1 + math.exp(-z * (x - center)))


def average_ragged_matrix(mat):
    return sum_ragged_matrix(mat) / count_ragged_matrix(mat)


def count_ragged_matrix(mat):
    return np.sum(np.fromiter((len(r) for r in mat), dtype=np.int32))


def sum_ragged_matrix(mat):
    return np.sum(np.fromiter((np.sum(r) for r in mat), dtype=np.float64))


def max_ragged_matrix(mat):
    return np.max([np.max(i) for i in mat])
