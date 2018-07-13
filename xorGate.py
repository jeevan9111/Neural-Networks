import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])
dataSet = np.ones((4, 4))
dataSet[:, 1:] = data[:, :]
weight = np.random.rand(3, 3)
weight_ = weight
lr = 0.3
j = 0
for i in range(1000000):

    print(i + 1, end="\t")
    zz = np.zeros(4, dtype='uint8')
    for index, row in enumerate(dataSet):
        y1 = np.matmul(row[:3], weight[0, :])
        z1 = sigmoid(y1)
        y2 = np.matmul(row[:3], weight[1, :])
        z2 = sigmoid(y2)
        row_ = np.array([1, y1, y2])
        y3 = np.matmul(row_, weight[2, :])
        z3 = sigmoid(y3)
        zz[index] = 0 if z3 < 0.5 else 1

        # start back progpagation
        delta = (z3 - row[3]) * z3 * (1 - z3)
        weight_[2] = weight[2] - lr * row_[0:3] * delta
        delta0 = delta * weight[2, 1] * z1 * (1 - z1)
        weight_[0] = weight[0] - lr * row[0:3] * delta0
        delta1 = delta * weight[2, 2] * z2 * (1 - z2)
        weight_[1] = weight[1] - lr * row[0:3] * delta1
        weight = weight_

    print(zz)

