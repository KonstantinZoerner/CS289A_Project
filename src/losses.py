from abc import ABC
import numpy as np

class Loss(ABC):
    pass

class One_Zero_Loss(Loss):
    def __call__(self, y, y_hat):
        # print(y.shape, y_hat.shape)
        # print(y, y_hat)
        y = y.reshape((-1,))
        y_hat = y_hat.reshape((-1,))
        return 1 - np.sum(y == y_hat) / y.size