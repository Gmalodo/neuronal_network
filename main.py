import gzip
from random import random, randint

import numpy as np
from numpy import argmax
from sklearn.datasets import make_blobs, make_circles, load_digits
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tqdm import tqdm

from neuron import predict_softmax, artificial_neuron_network
from views import image, learning_stats, accuracy


def training_images():
    with gzip.open('training_set/train-images-idx3-ubyte.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        return images / images.max()


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
    # training_set = training_images()
    # training_labels = training_labels()
    # x = training_set.T.reshape(training_set.shape[2] * training_set.shape[1], training_set.shape[0])
    # y = np.expand_dims(training_labels, axis=0)
    training_set = load_digits()
    x = MinMaxScaler().fit_transform(training_set.data)
    y = LabelBinarizer().fit_transform(training_set.target)
    W, b, loss, acc = artificial_neuron_network(x.T, y.T, [32, 32], 10000, y_o=training_set.target)
    learning_stats(loss)
    # cost(x.T[:10000].T, LabelBinarizer().fit_transform(y[0])[:10000])
    # image(x, y, W, b)
    accuracy(acc)
    print(training_set.target[:10])
    print("result", predict_softmax(x.T, W, b)[:10])
