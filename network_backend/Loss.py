import numpy as np


# interface for loss functions
class LossI:
    def loss(self, out, labels):
        raise NotImplementedError()

    def d(self, out, labels):
        raise NotImplementedError()

    # return is loss, delta
    def __call__(self, out, labels):
        return self.loss(out, labels), self.d(out, labels)


class L2Loss(LossI):
    def loss(self, out, labels):
        return 1/2 * np.outer(labels-out, labels-out)

    def d(self, out, labels):
        return np.outer(out - labels, np.ones(labels.shape))[0]


class BCELoss(LossI):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def loss(self, out, labels):
        return -(labels * np.log2(out+self.eps) + (1-labels) * np.log2(1-out+self.eps))

    def d(self, out, labels):
        return (1-labels)/(1-out+self.eps) - labels/(out+self.eps)
