from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction:
    @abstractmethod
    def activate(self, x):
        """
        Compute activation for input x.
        """
        pass

    @abstractmethod
    def backward(self, z, grads):
        """
        Compute the backward update of the activation function for input x.
        """
        pass

class Linear(ActivationFunction):
    """
    f(x) = x
    """
    def __init__(self):
        pass

    def activate(self, x):
        """
        f(x) = x
        """
        return x
    
    def backward(self, z, grads):
        """
        f'(x) = 1
        """
        return grads

class ReLU(ActivationFunction):
    """
    f(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()

    def activate(self, x):
        """
        f(x) = max(0, x)
        """
        return np.where(x > 0, x, 0)
    
    def backward(self, z, grads):
        """
        f'(x) = switch:
            case x > 0: 1
            else:       0
        """
        return grads * np.where(z > 0, 1, 0)
    
class Sigmoid(ActivationFunction):
    """
    f(x) = 1 / (1 + e^{-x})
    """
    def __init__(self):
        pass

    def activate(self, x):
        """
        f(x) = 1 / (1 + e^{-x})
        """
        return 1 / (1 + np.exp(-x))
    
    def backward(self, z, grads):
        """
        f'(x) = e^{-x} / (1 + e^{-x})^2 = 1 / (1 + e^{-x}) - 1 / (1 + e^{-x})^2 = 1 / (1 + e^{-x})[1 - 1 / (1 + e^{-x})]
        """
        s = self.activate(z)
        return grads * s * (1 - s)
    
class Softmax(ActivationFunction):
    def __init__(self):
        pass

    def activate(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, z, grads):
        s = self.activate(z)
        dot = np.sum(s * grads, axis=1, keepdims=True)
        return s * grads - s * dot