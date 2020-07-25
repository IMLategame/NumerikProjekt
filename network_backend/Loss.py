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
        outer = 1/2 * np.outer(labels-out, labels-out)
        return [outer[i][i] for i in range(outer.shape[0])]

    def d(self, out, labels):
        outer_prod = np.outer(out - labels, np.ones(labels.shape))
        return np.array([row[0] for row in outer_prod])[np.newaxis, :]


class BCELoss(LossI):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def loss(self, out, labels):
        return -(labels * np.log(out+self.eps) + (1-labels) * np.log(1-out+self.eps))

    def d(self, out, labels):
        return (1 - labels) / (1 - out + self.eps) - labels / (out + self.eps)


class CrossEntropyLoss(LossI):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def loss(self, out, labels):
        return -np.sum(labels * np.log(out), axis=0)

    def d(self, out, labels):
        return -labels/(out + self.eps)


class SplitLosses(LossI):
    def __init__(self, sizes, losses):
        assert len(sizes) > 0
        assert len(sizes) == len(losses)
        for loss in losses:
            assert isinstance(loss, LossI)
        self.sizes = sizes
        self.split_sizes = [sum(self.sizes[:i + 1]) for i in range(len(self.sizes))]
        self.losses = losses

    def loss(self, out, labels):
        split_out = np.split(out, self.split_sizes, axis=0)
        split_labels = np.split(labels, self.split_sizes, axis=0)
        split_out = [loss.loss(out, labels) for loss, out, labels in zip(self.losses, split_out, split_labels)]
        return np.concatenate(split_out, axis=0)

    def d(self, out, labels):
        split_out = np.split(out, self.split_sizes, axis=0)
        split_labels = np.split(labels, self.split_sizes, axis=0)
        split_out = [loss.d(out, labels) for loss, out, labels in zip(self.losses, split_out, split_labels)]
        return np.concatenate(split_out, axis=0)
