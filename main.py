import gzip
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, load_digits
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from neuron import predict_softmax, artificial_neuron_network
from views import learning_stats, accuracy


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
    training_labels = training_labels()
    x = training_set.T.reshape(training_set.shape[2] * training_set.shape[1], training_set.shape[0])
    y = LabelBinarizer().fit_transform(training_labels)
    print(y)
    # training_set = load_digits()
    # x = training_set.data
    # y = np.expand_dims(training_set.target, axis=0) + 1 / 10
    # y = np.expand_dims(training_set.target, axis=0) + 1 / 10
    W, b, loss, acc = artificial_neuron_network(x, y.T, [128, 128, 128, 128], 100, y_o=training_set.T)
    learning_stats(loss)
    accuracy(acc)
    # print((predict_softmax(1, W, b).T[0] * 256).T.shape)
    plt.figure()
    plt.imshow((predict_softmax(y[1], W, b).T[0] * 256).T.reshape(8, 8))
    # fig, axs = plt.subplots(16, 16)
    # k = 0
    # for i in range(len(axs)):
    #     for j in range(len(axs[i])):
    #         axs[i][j].imshow((predict_softmax(y[0], W, b).T[k] * 256).reshape(8, 8))
    #         k = k + 1
    plt.show()
