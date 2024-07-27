import numpy as np


class ReLU:
    def forward(self, z):
        self.z = z
        self.a = np.maximum(0, z)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[dvalues <= 0] = 0


class Softmax:
    def forward(self, z):
        exponentials = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.a = exponentials / np.sum(exponentials, axis=1, keepdims=True)
