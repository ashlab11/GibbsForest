import numpy as np
import random
from abc import ABC, abstractmethod

class BaseLoss(ABC):
    """
    Abstract base class for loss functions.
    Each loss function should implement function, gradient, hessian, and the argmin function.
    """

    @abstractmethod
    def __call__(self, y_true, y_pred):
        """
        Computes the loss function.
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted values

        Returns:
        - Loss value
        """
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        """
        Computes the first derivative of the loss function.
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted values

        Returns:
        - Gradient of the loss function
        """
        pass

    @abstractmethod
    def hessian(self, y_true, y_pred):
        """
        Computes the second derivative (Hessian) of the loss function.
        
        Parameters:
        - y_true: True labels
        - y_pred: Predicted values

        Returns:
        - Hessian of the loss function
        """
        pass

    @abstractmethod
    def argmin(self, y_true):
        """
        Function that calculates the argmin given y values.
        
        Parameters:
        - y_true: True labels

        Returns:
        - Constant argmin value
        """
        pass
    

class LeastSquaresLoss(BaseLoss):
    """
    Least Squares Loss function.
    """

    def __call__(self, y_true, y_pred):
        return 0.5 * np.sum((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        return y_pred - y_true

    def hessian(self, y_true, y_pred):
        return np.ones_like(y_true)

    def argmin(self, y_true):
        return np.mean(y_true)
    
class LeastAbsoluteLoss(BaseLoss):
    """
    Least Absolute Loss function.
    """

    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true)

    def hessian(self, y_true, y_pred):
        return np.zeros(len(y_true))

    def argmin(self, y_true):
        return np.median(y_true)
    
class CrossEntropyLoss(BaseLoss):
    """
    Cross Entropy Loss function.
    """

    def __call__(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def hessian(self, y_true, y_pred):
        """TODO: Implement this!"""
        return y_pred * (1 - y_pred)

    def argmin(self, y_true):
        return np.mean(y_true)