import numpy as np


class ActivationI:
    def call(self, x):
        """
            :param x: x
            :return: f(x)
        """
        raise NotImplementedError()

    def d(self, y):
        """
            :param y: y = f(x)
            :return: f'(x)
        """
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


class Sigmoid(ActivationI):
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


class ReLU(ActivationI):
    def __init__(self):
        self.f = np.vectorize(elementWiseReLU)
        self.df = np.vectorize(elementWiseReLUDer)

    def call(self, x):
        return self.f(x)

    def d(self, x):
        return self.df(x)


class LeakyReLU(ActivationI):
    def __init__(self, rate=0.1):
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
        obj = super(LeakyReLU, self).toDict()
        obj["rate"] = self.rate
        return obj

    @classmethod
    def fromDict(cls, obj):
        if "rate" not in obj:
            obj["rate"] = 0.1
        return fctn_dict[obj["name"]](obj["rate"])


class Identity(ActivationI):
    def call(self, x):
        return x

    def d(self, x):
        return np.ones_like(x)


class Tanh(ActivationI):
    def call(self, x):
        return np.tanh(x)
        
    def d(self, x):
        return 1-x*x


class Softmax(ActivationI):
    """
        ONLY USE AS OUTPUT INTO CROSS ENTROPY LOSS
    """
    def call(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

    def d(self, x):
        return np.ones_like(x)


fctn_dict = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "Identity": Identity,
    "Tanh": Tanh,
    "Softmax": Softmax,
    "LeakyReLU": LeakyReLU,
    "LeakyReLu": LeakyReLU
}
