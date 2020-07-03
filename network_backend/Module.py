class ModuleI:

    # call super().__init__() in subclasses
    def __init__(self):
        self.mode_train = True

    # x.shape(batch_size, ...)
    def feed_forward(self, x):
        raise NotImplementedError()

    # delta_out -> delta_in, [gradients]
    def backprop(self, delta_out):
        raise NotImplementedError()

    # weights += delta
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

    def fromString(self, string):
        raise NotImplementedError()