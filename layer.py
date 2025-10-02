import numpy as np
from activation import ActivationFunction
from activation import Linear

class Dense():
    def __init__(self, input_shape, output_shape, activation: ActivationFunction = Linear):
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape))
        self.biases = np.zeros(output_shape)
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.z = np.matmul(self.x, self.weights) + self.biases
        return self.activation.activate(self.z)
    
    def backward(self, grads):
        grad_z = self.activation.backward(self.z, grads)
        self.grad_weights = np.dot(self.x.T, grad_z)
        self.grad_biases = np.sum(grad_z, axis=0)
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    