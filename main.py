from sklearn.datasets import make_blobs

from neuron import Neuron


def make_dataset():
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    return X, y


if __name__ == '__main__':
    x, y = make_dataset()
    n = Neuron()
    n.artificial_neuron(x, y)
    # nbr_neuron = [3, 1]
    # X = []
    # Y = [[y]]
    # for index in range(len(nbr_neuron)):
    #     X.append([])
    #     for index2 in range(nbr_neuron[index]):
    #         n = Neuron()
    #         if index == 0:
    #             X[0].append(x)
    #             r, b1 = n.artificial_neuron(X[0][index2], y)
    #         else:
    #             print(X[1])
    #             r, b1 = n.artificial_neuron(X[index][index2], y, view="sigmoid")
    #         X.append([])
    #         X[index + 1].append(r)
