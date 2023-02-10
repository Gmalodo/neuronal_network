import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go


class Views:
    @staticmethod
    def learning_stats(Loss):
        plt.plot(Loss)
        plt.show()

    @staticmethod
    def decisions_frontier(X, W, b, y):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')

        x1 = np.linspace(-1, 4, 100)
        x2 = (- W[0] * x1 - b) / W[1]

        ax.plot(x1, x2, c='orange', lw=3)

    @staticmethod
    def decision_sigmoid_3D(X, W, b, y):
        X0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        X1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        xx0, xx1 = np.meshgrid(X0, X1)
        Z = W[0] * xx0 + W[1] * xx1 + b
        A = 1 / (1 + np.exp(-Z))

        fig = (go.Figure(data=[go.Surface(z=A, x=xx0, y=xx1, colorscale='YlGn', opacity=0.7, reversescale=True)]))

        fig.add_scatter3d(x=X[:, 0].flatten(), y=X[:, 1].flatten(), z=y.flatten(), mode='markers',
                          marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity=0.9, reversescale=True))

        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.show()

    @staticmethod
    def pol_decision_frontier(X, y, W, b):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(X[0, :], X[1, :], c=y, cmap='summer')

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                             np.arange(y_min, y_max, 0.2))
        x1 = np.linspace(x_max - x_min, y_min - y_max, 100)
        # x2 = (- W[0] * xx - b[0]) / W[1]
        # x2 = (- W[0] * xx - b[0]) / W[1]
        x = []
        # for layer in range(len(params)):
        #     x.append([xx])
        #     for i in range(len(params[layer])):
        #         print(params[layer][i]["W"][0] * x[layer][0])
        #         x[layer].append((- params[layer][i]["W"][0] * x[layer][0] - params[layer][i]["b"][0])
        #                         / params[layer][i]["W"][1])

        cp = plt.contour(xx, yy, x[-1], cmap=plt.cm.Paired)
        plt.colorbar(cp)
        plt.show()

    @staticmethod
    def accuracy(accuracy):
        _, ax = plt.subplots(figsize=(9, 6))
        ax.plot(accuracy)
        plt.show()
