import numpy as np

class ModuleI:

    # call super().__init__() in subclasses
    def __init__(self, str=None):
        if str is not None:
            self.fromString(str)
        self.mode_train = True

    # for nicer syntax
    def __call__(self, x):
        return self.feed_forward(x)

    # for nicer syntax
    def __len__(self):
        return self.noFeatures()

    # for nicer syntax
    def __str__(self):
        return self.toString()

    # for nicer syntax
    def __repr__(self):
        return self.toString()

    # for nicer syntax
    def __getitem__(self, item):
        assert item >= 0
        assert item < len(self)
        return self.getGradients()[item]

    def toFile(self, f):
        f.write(self.toString())

    # TODO: Implement
    @classmethod
    def fromFile(cls, f): ...

    # x.shape(batch_size, ...)
    def feed_forward(self, x):
        raise NotImplementedError()

    # delta_out -> delta_in, [gradients]
    def backprop(self, delta_out):
        raise NotImplementedError()

    # weights -= delta
    def update(self, delta):
        raise NotImplementedError()

    # len([gradients])
    def noFeatures(self):
        raise NotImplementedError()

    # can use self.mode_train in layers. Needed for Dropout, etc.
    def set_mode(self, mode):
        if mode in ['train', 'Train']:
            self.mode_train = True
        elif mode in ['test', 'Test', 'eval', 'Eval']:
            self.mode_train = False

    def getGradients(self):
        raise NotImplementedError()

    def toString(self):
        raise NotImplementedError()

