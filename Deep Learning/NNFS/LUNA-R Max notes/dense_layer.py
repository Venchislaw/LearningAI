import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data


class Dense:
    def __init__(self, n_inputs, n_neurons, weights=None) -> None:
        if not weights:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        else:
            self.weights = weights
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.z


X, y = spiral_data(samples=100, classes=2)
z = Dense(2, 5).forward(X)
print(z[:5], z.shape)
