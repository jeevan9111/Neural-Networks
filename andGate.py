import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


data = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
])
dataSet = np.ones((4, 4))
dataSet[:, 1:] = data[:, :]
weight = np.array([[0.5, 0.5, 0.7],
                   [0.6, 0.4, 0.6],
                   [0.4, 0.5, 0.6]])
weight_ = weight
j = 0
for i in range(2000000):
    for row in dataSet:
        y1 = np.matmul(row[:3], weight[0, :])
        z1 = sigmoid(y1)
        y2 = np.matmul(row[:3], weight[1, :])
        z2 = sigmoid(y2)
        row_ = np.array([1, y1, y2])
        y3 = np.matmul(row_, weight[2, :])
        z3 = sigmoid(y3)
        print(z3, end='\t\t')
        delta = (z3 - row[3]) * z3 * (1 - z3)
        weight_[2] = weight[2] - 0.3 * row_[0:3] * delta
        delta0 = delta * weight[2, 1] * z1 * (1 - z1)
        weight_[0] = weight[0] - 0.3 * row[0:3] * delta0
        delta1 = delta * weight[2, 2] * z2 * (1 - z2)
        weight_[1] = weight[1] - 0.3 * row[0:3] * delta1
        weight = weight_
    print('\n')
print(weight)
