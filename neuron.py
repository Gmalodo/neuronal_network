import numpy as np
from sklearn.metrics import accuracy_score

from views import Views


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W, b


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return dW, db


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b


def update_network(dW, db, W, b, learning_rate):
    for c in range(len(W)):
        W[c] = W[c] - learning_rate * dW[c]
        b[c] = b[c] - learning_rate * db[c]
    return W, b


def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


def init_network(layers):
    W = []
    b = []
    for c in range(1, len(layers)):
        W.append(np.random.rand(layers[c], layers[c - 1]))
        b.append(np.random.rand(layers[c], 1))

    return W, b


def forward(X, W, b):
    A = [X]
    for c in range(len(W)):
        Z = W[c].dot(A[c]) + b[c]
        A.append(1 / (1 + np.exp(-Z)))
    return A


def backward(A, y, W):
    dZ = A[-1] - y
    dW = []
    db = []

    for c in reversed(range(len(W))):
        dW.append(1 / y.shape[1] * np.dot(dZ, A[c].T))
        db.append(1 / y.shape[1] * np.sum(dZ, axis=1, keepdims=True))
        if c > 1:
            dZ = np.dot(W[c].T, dZ) * A[c - 1] * (1 - A[c - 1])
    return dW[::-1], db[::-1]


def predict_network(x, W, b):
    A = forward(x, W, b)
    return A[-1] >= 0.5


class Neuron:

    @staticmethod
    def artificial_neuron(X, y, learning_rate=0.1, n_iter=100, view=''):
        W, b = initialisation(X)

        Loss = []

        for i in range(n_iter):
            A = model(X, W, b)
            Loss.append(log_loss(A, y))
            dW, db = gradients(A, X, y)
            W, b = update(dW, db, W, b, learning_rate)

        if view == 'Loss':
            Views.learning_stats(Loss)
        if view == 'frontier':
            Views.decisions_frontier(X, W, b, y)
        if view == 'sigmoid':
            Views.decision_sigmoid_3D(X, W, b, y)

    @staticmethod
    def artificial_neuron_network(x, y, hidden_layers, iterations, learning_rate=0.1):
        Loss = []
        acc = []
        layers = hidden_layers
        layers.insert(0, x.shape[0])
        layers.append(y.shape[0])
        W, b = init_network(layers)
        for i in range(iterations):
            A = forward(x, W, b)
            dW, db = backward(A, y, W)
            W, b = update_network(dW, db, W, b, learning_rate)
            if i % 10 == 0:
                Loss.append(log_loss(A[-1], y))
                y_predict = predict_network(x, W, b)
                acc.append(accuracy_score(y.flatten(), y_predict.flatten()))

        Views.learning_stats(Loss)
        Views.accuracy(acc)
        Views.pol_decision_frontier(x, y, W, b)
