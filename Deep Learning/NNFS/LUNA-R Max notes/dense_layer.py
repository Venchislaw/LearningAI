import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from activations_funcs import ReLU, Softmax
from losses import CategoricalCrossentropy


class Dense:
    def __init__(self, n_inputs, n_neurons, weights=None) -> None:
        if not weights:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        else:
            self.weights = weights
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 3)
activation1 = ReLU()
dense2 = Dense(3, 3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.z)
dense2.forward(activation1.a)
activation2.forward(dense2.z)


loss = CategoricalCrossentropy()
print(loss.calculate(y, activation2.a))