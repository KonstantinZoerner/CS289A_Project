from abc import ABC
import numpy as np

class Loss(ABC):
    pass

class One_Zero_Loss(Loss):
    def __call__(self, y, y_hat):
        return np.sum(y == y_hat) / y.size