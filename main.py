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

    # x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    # y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
    #                      np.arange(y_min, y_max, 0.2))
    # n = Neuron()
    # A1, W1 = n.artificial_neuron(x, y)
    # print(x2.shape)
    n2 = Neuron()
    n2.artificial_neuron_network(x, y, [30, 1])
    # plt.contour(xx, yy, , cmap=plt.cm.Paired)
    # plt.show()