import numpy as np

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
    W1 = W[0] - learning_rate * dW[0]
    b1 = b[0] - learning_rate * db[0]
    W2 = W[1] - learning_rate * dW[1]
    b2 = b[1] - learning_rate * db[1]
    return (W1, W2), (b1, b2)


def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


def init_network(entry, layer1, layer2):
    if layer2 is not None:
        W1 = np.random.rand(layer1, entry)
        b1 = np.random.rand(layer1, 1)
        W2 = np.random.rand(layer2, layer1)
        b2 = np.random.rand(layer2, 1)
        return (W1, W2), (b1, b2)
    return None, None


def forward(X, W, b):
    Z1 = W[0].dot(X) + b[0]
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W[1].dot(A1) + b[1]
    A2 = 1 / (1 + np.exp(-Z2))
    return A1, A2


def backward(A, X, y, W):
    dZ2 = A[1] - y
    dW2 = 1 / y.shape[1] * dZ2.dot(A[0].T)
    db2 = 1 / y.shape[1] * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W.T, dZ2) * A[0] * (1 - A[0])
    dW1 = 1 / y.shape[1] * dZ1.dot(X.T)
    db1 = 1 / y.shape[1] * np.sum(dZ1, axis=1, keepdims=True)

    return (dW1, dW2), (db1, db2)


def predict_network(x, W, b):
    A = forward(x, W, b)
    return A[1] >= 0.5


class Neuron:

    def artificial_neuron(self, X, y, learning_rate=0.1, n_iter=100, view=''):
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

    def artificial_neuron_network(self, x1, y, layer, learning_rate=0.1):
        global W, b
        params = []
        x = x1
        for i in range(len(layer)):
            params.append([])
            entry = x.shape[0]
            n2 = y.shape[0]
            W, b = init_network(entry, layer[i], n2)
            Loss = []
            if W is not None:
                for j in range(1000):
                    A = forward(x, W, b)
                    dW, db = backward(A, x, y, W[1])
                    W, b = update_network(dW, db, W, b, learning_rate)
                    if j % 10 == 0:
                        Loss.append(log_loss(A[1], y))
            # params[i].append({"W": W, "b": b})
            x = A[1]
            # Views.pol_decision_frontier(x1, y, params)