from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def step(self, model):
        """
        Update model parameters using computed gradients
        """
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model):
        for layer in model.layers:
            layer.weights -= self.learning_rate * layer.grad_weights
            layer.biases -= self.learning_rate * layer.grad_biases