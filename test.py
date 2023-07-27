import numpy as np


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


y = np.array([3, 2, 1, 3, 0])
print(y.shape)
print(y.reshape(-1).shape)

C = 4
new_y = convert_to_one_hot(y, C)
print(new_y)
print(type(new_y))
