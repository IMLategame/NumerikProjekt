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
        return 1/2*(labels-out).T @ (labels-out)

    def d(self, out, labels):
        return (out - labels).T @ np.ones(labels.shape)


class BCELoss(LossI):
    def loss(self, out, labels):
        return -(labels * np.log2(out) + (1-labels) * np.log2(1-out))

    def d(self, out, labels):
        return (1-labels)/(1-out) - labels/out
