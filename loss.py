from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def loss(self, pred, target):
        """
        Compute the loss value given predictions and targets
        """
        pass

    @abstractmethod
    def derivative(self, pred, target):
        """
        Compute the gradient of the loss from predictions
        """
        pass

class MeanAbsoluteError(LossFunction):
    def loss(self, pred, target):
        """
        f(y, y') = abs(y - y') / n
        """
        return np.mean(np.abs(pred - target))
    
    def derivative(self, pred, target):
        """
        f(y, y') = switch:
            case y - y' > 0: 1 / n
            case y - y' < 0: -1 / n
            case y - y' = 0: 0
        """
        return np.sign(pred - target) / pred.size
    
class MeanSquareError(LossFunction):
    def loss(self, pred, target):
        """
        f(y, y') = (y - y')^2 / n
        """
        return np.mean((pred - target) ** 2)
    
    def derivative(self, pred, target):
        """
        f(y, y') = 2(y - y') / n
        """
        return (2 * (pred - target)) / pred.size
    

class SparseCategoricalCrossentropy(LossFunction):
    def loss(self, pred, target):
        pred = np.clip(pred, 1e-15, 1.0 - 1e-15)
        target = target.squeeze(-1)
        batch_size = pred.shape[0]
        log_probs = -np.log(pred[np.arange(batch_size), target])
        return np.mean(log_probs)
    
    def derivative(self, pred, target):
        pred = np.clip(pred, 1e-15, 1.0 - 1e-15)
        target = target.squeeze(-1)
        batch_size, _ = pred.shape
        grad = np.zeros_like(pred)
        grad[np.arange(batch_size), target] = -1 / pred[np.arange(batch_size), target]
        return grad / batch_size