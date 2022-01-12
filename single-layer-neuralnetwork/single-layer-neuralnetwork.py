import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from scipy.special import softmax

class SingleNeuralNetwork():
    def __init__(self, X_train, Y_train, nodes):
        classes = len(np.unique(Y_train))
        self.X = X_train
        self.Y = Y_train.reshape(-1, 1)
        self.W1 = np.random.randn(nodes, X_train.shape[1]) * 0.01
        self.b1 = np.zeros((nodes, 1))
        self.W2 = np.random.randn(classes, nodes)
        self.b2 = np.zeros((classes, 1))

    def relu(self, z):
        return np.maximum(z, 0)

    # def softmax(self, z):
    #     e_x = np.exp(z - np.max(z))
    #     return e_x / e_x.sum()

    def forward(self, X, W1, W2, b1, b2):
        self.Z1 = X @ W1.T + b1.T
        self.a1 = self.relu(self.Z1)
        self.Z2 = self.a1 @ W2.T + b2.T
        self.a2 = softmax(self.Z2)
        return

    def cost_function(self, y, a2):
        m = y.shape[0]
        temp = np.multiply(y, np.log(a2)) + np.multiply((1-y), np.log(1-a2))
        cost = (-1/m) * np.sum(temp)
        return cost

    def gredient(self, x, y, a1, a2, W2):
        m = y.shape[0]
        dz2 = a2 - y
        self.dW2 = (1/m) * dz2.T @ a1
        self.db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True).T
        dz1 = (dz2 @ W2) * (1-np.power(a1, 2))
        self.dW1 = (1/m) * dz1.T @ x
        self.db1 = (1/m) * (np.sum(dz1, axis=0, keepdims=True)).T
        return

    def back_propagation(self, alpha):
        self.W1 -= alpha * self.dW1
        self.b1 -= alpha * self.db1
        self.W2 -= alpha * self.dW2
        self.b2 -= alpha * self.db2

    def fit(self, iterations, lr):
        cost = []
        for _ in range(iterations):
            self.forward(self.X, self.W1, self.W2, self.b1, self.b2)
            cost.append(self.cost_function(self.Y, self.a2))
            self.gredient(self.X, self.Y, self.a1, self.a2, self.W2)
            self.back_propagation(lr)
        return print(cost)

X = load_iris().data[:100]
Y = load_iris().target[:100]
x_train_splited, x_test_splited, y_train_splited, y_test_splited = train_test_split(X, Y)

model = SingleNeuralNetwork(x_train_splited, y_train_splited, 5)
model.fit(20, 0.005)