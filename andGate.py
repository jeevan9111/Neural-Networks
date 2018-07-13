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
weight = np.random.rand(3)
print("No \t Output \t \t\tWeights")
for i in range(100):
    print(i + 1, end="\t")
    zz = np.zeros(4, dtype='uint8')
    for index, row in enumerate(dataSet):
        y = np.matmul(row[:3], weight)
        z = sigmoid(y)
        zz[index] = 0 if z < 0.5 else 1

        delta = (z - row[3])
        a = z * (1 - z)
        weight = weight - 0.3 * row[0:3] * a * delta
    print(zz, weight)
