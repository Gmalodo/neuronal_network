import gzip

import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import LabelBinarizer

from neuron import Neuron


def training_images():
    with gzip.open('training_set/train-images-idx3-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        return images


def training_labels():
    with gzip.open('training_set/train-labels-idx1-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

def make_dataset():
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    return X, y


if __name__ == '__main__':
    # x, y = make_blobs(n_samples=100, n_features=2, centers=3, random_state=0)
    # x = x.T
    # y = LabelBinarizer().fit_transform(y).T

    x = training_images().reshape(784, 60000)
    y = np.expand_dims(training_labels(), axis=0)
    # y = np.expand_dims(y, axis=0)
    print(y[0].shape)
    n2 = Neuron()
    n2.artificial_neuron_network(x, LabelBinarizer().fit_transform(y[0]).T, y, [32, 32], 100000)
