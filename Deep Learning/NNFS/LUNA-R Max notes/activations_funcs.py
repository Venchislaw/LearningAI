import numpy as np


class ReLU:
    def output(self, z):
        return np.maximum(z, 0)


class Softmax:
    def output(self, z):
        exponentials = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)
    
print(Softmax().output([[1, 2, 3]]))