import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from views import learning_stats, decisions_frontier, decision_sigmoid_3D


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


def log_loss_network(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A[-1]) - (1 - y) * np.log(1 - A[-1]))


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
        W.append(np.random.randn(layers[c], layers[c - 1]))
        b.append(np.random.randn(layers[c], 1))
    return W, b


def forward(X, W, b):
    A = [X]
    for c in range(len(W) - 1):
        Z = W[c].dot(A[c]) + b[c]
        A.append(1 / (1 + np.exp(-Z)))
    Z = W[-1].dot(A[-1]) + b[-1]
    A.append(np.exp(Z) / np.sum(np.exp(Z), axis=0))
    return A


def backward(A, y, W):
    dZ = A[-1] - y
    dW = []
    db = []
    for c in reversed(range(len(W))):
        dW.append(1 / y.shape[1] * np.dot(dZ, A[c].T))
        db.append(1 / y.shape[1] * np.sum(dZ, axis=1, keepdims=True))
        dZ = np.dot(W[c].T, dZ) * (A[c] * (1 - A[c]))
    return dW[::-1], db[::-1]


def predict_network(x, W, b):
    A = forward(x, W, b)
    return A[-1] >= 0.5


def predict_softmax(x, W, b):
    A = forward(x, W, b)
    return argmax(A[-1], axis=0)


def artificial_neuron(X, y, learning_rate=0.1, n_iter=100, view=''):
    W, b = initialisation(X)

    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    if view == 'Loss':
        learning_stats(Loss)
    if view == 'frontier':
        decisions_frontier(X, W, b, y)
    if view == 'sigmoid':
        decision_sigmoid_3D(X, W, b, y)


def artificial_neuron_network(x, y, hidden_layers, iterations, learning_rate=0.1, y_o=""):
    Loss = []
    acc = []
    Wtemp = []
    layers = hidden_layers
    layers.insert(0, x.shape[0])
    layers.append(y.shape[0])
    W, b = init_network(layers)
    print("learning")
    for i in tqdm(range(iterations)):
        A = forward(x, W, b)
        dW, db = backward(A, y, W)
        W, b = update_network(dW, db, W, b, learning_rate)
        Wtemp.append(W[-1].max())
        if i % 10 == 0:
            Loss.append(log_loss_network(A, y))
            acc.append(accuracy_score(y_o, predict_softmax(x, W, b)))
    plt.figure()
    plt.plot(Wtemp)
    plt.show()
    return W, b, Loss, acc
