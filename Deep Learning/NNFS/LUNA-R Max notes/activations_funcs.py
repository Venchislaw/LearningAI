import numpy as np


class ReLU:
    def forward(self, z):
        self.a = np.maximum(0, z)


class Softmax:
    def forward(self, z):
        exponentials = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.a = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    
print(Softmax().forward([[1, 2, 3]]))
