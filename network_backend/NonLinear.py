import numpy as np

class NonLinearI:
    def call(self, x):
        raise NotImplementedError()

    def d(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.call(x)

class Sigmoid(NonLinearI):
    def call(self, x):
        return 1/(1 + np.exp(-x))

    def d(self, x):
        return self.call(x) * (1 - self.call(x))