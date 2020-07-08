import numpy as np

class NonLinearI:
    def call(self, x):
        raise NotImplementedError()

    # Watch out. This is NOT the derivative, but the derivative after applying the function:
    # d = f' o f^-1
    def d(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.call(x)

class Sigmoid(NonLinearI):
    def call(self, x):
        return 1/(1 + np.exp(-x))

    def d(self, x):
        return x * (1-x) #self.call(x) * (1 - self.call(x))

def elementWiseReLU(x):
    return max(x, 0)

def elementWiseReLUDer(x):
    if x > 0:
        return 1
    return 0

class ReLU(NonLinearI):
    def __init__(self):
        self.f = np.vectorize(elementWiseReLU)
        self.df = np.vectorize(elementWiseReLUDer)

    def call(self, x):
        return self.f(x)

    def d(self, x):
        return self.df(x)
