import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles

from neuron import Neuron


def make_dataset():
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    return X, y


if __name__ == '__main__':
    x, y = make_dataset()

    n2 = Neuron()
    n2.artificial_neuron_network(x, y, [32, 32], 1000)
