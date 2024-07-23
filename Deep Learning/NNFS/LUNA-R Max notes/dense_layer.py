import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons, weights=None) -> None:
        self.weights = np.random.rand((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        pass