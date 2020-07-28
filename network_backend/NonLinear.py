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

    def toDict(self):
        obj = {"name": type(self).__name__}
        return obj

    @classmethod
    def fromDict(cls, obj):
        # for backwards compatibility
        if isinstance(obj, str):
            return fctn_dict[obj]()
        return fctn_dict[obj["name"]]()


class Sigmoid(NonLinearI):
    def call(self, x):
        return 1/(1 + np.exp(-x))

    def d(self, x):
        return x * (1 - x)


def elementWiseReLU(x, alpha=0.0):
    return max(x, alpha*x)


def elementWiseReLUDer(x, alpha=0.0):
    if x > 0:
        return 1.0
    return alpha


class ReLU(NonLinearI):
    def __init__(self):
        self.f = np.vectorize(elementWiseReLU)
        self.df = np.vectorize(elementWiseReLUDer)

    def call(self, x):
        return self.f(x)

    def d(self, x):
        return self.df(x)


class LeakyReLu(NonLinearI):
    def __init__(self, rate):
        assert rate != 1.0
        self.rate = rate

        def func(x):
            return elementWiseReLU(x, rate)
        self.f = np.vectorize(func)

        def deriv(x):
            return elementWiseReLUDer(x, rate)
        self.df = np.vectorize(deriv)

    def call(self, x):
        return self.f(x)

    def d(self, x):
        return self.df(x)

    def toDict(self):
        obj = super(LeakyReLu, self).toDict()
        obj["rate"] = self.rate

    @classmethod
    def fromDict(cls, obj):
        if "rate" not in obj:
            obj["rate"] = 0.1
        return fctn_dict[obj["name"]](obj["rate"])


class Identity(NonLinearI):
    def call(self, x):
        return x

    def d(self, x):
        return np.ones_like(x)


class Tanh(NonLinearI):
    def call(self, x):
        return np.tanh(x)
        
    def d(self, x):
        return 1-x*x


class Softmax(NonLinearI):
    def call(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def d(self, x):
        return x * (1 - x)


fctn_dict = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "Identity": Identity,
    "Tanh": Tanh,
    "Softmax": Softmax,
    "LeakyReLu": LeakyReLu
}
