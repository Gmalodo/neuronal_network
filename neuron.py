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


def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


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

        return W, b
