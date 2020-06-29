class ModuleI:

    # x.shape(batch_size, ...)
    def feed_forward(self, x):
        raise NotImplementedError()

    # delta_out -> delta_in, [gradients]
    def backprop(self, delta_out):
        raise NotImplementedError()

    # weights += lr * gradients
    def update(self, lr, gradients):
        raise NotImplementedError()

    # len([gradients])
    def noFeatures(self):
        raise NotImplementedError()

    def getGradients(self):
        raise NotImplementedError()

    def toString(self):
        raise NotImplementedError()

    def fromString(self, string):
        raise NotImplementedError()