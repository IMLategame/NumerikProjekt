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
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]
        outer = 1/2 * (labels-out).T@(labels-out)
        return np.array([outer[i][i] for i in range(outer.shape[0])])

    def d(self, out, labels):
        return out-labels


class BCELoss(LossI):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def loss(self, out, labels):
        return -(labels * np.log(out+self.eps) + (1-labels) * np.log(1-out+self.eps))

    def d(self, out, labels):
        return (1 - labels) / (1 - out + self.eps) - labels / (out + self.eps)


class CrossEntropyLoss(LossI):
    """
        ONLY USE IF LAST LAYER IS SOFTMAX LAYER
    """
    def __init__(self, eps=1e-10):
        self.eps = eps

    def loss(self, out, labels):
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]
        return -np.sum(labels * np.log(out + self.eps), axis=0)

    def d(self, out, labels):
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]
        return out-labels


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
