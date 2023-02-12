import gzip

import numpy as np
from numpy import argmax
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tqdm import tqdm

from neuron import predict_softmax, artificial_neuron_network
from views import image, learning_stats, cost


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
    training_set = training_images()
    x = training_set.T.reshape(training_set.shape[2] * training_set.shape[1], training_set.shape[0])
    y = np.expand_dims(training_labels(), axis=0)
    W, b, loss = artificial_neuron_network(x.T[:100].T, LabelBinarizer().fit_transform(y[0])[:100].T, [32, 32], 10000)
    learning_stats(loss)
    # cost(x.T[:10000].T, LabelBinarizer().fit_transform(y[0])[:10000])
    image(x, y, W, b)
    print("predict", y[0][:10])
    print("result", predict_softmax(x.T[:10].T, W, b))
