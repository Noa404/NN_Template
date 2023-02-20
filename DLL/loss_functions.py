"""
A loss functino measures how good our prediictions are
"""
from noahnet.tensor import Tensor

import numpy as np


class Loss:
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE is mean squared error,
    although were just going to do total squared error

    """
    def loss(self, predicted: Tensor,actual: Tensor) -> float:
        return np.sum((predicted - actual) **2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

