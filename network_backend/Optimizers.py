class OptimizerI:
    def get_rate(self):
        raise NotImplementedError()

class SGD(OptimizerI):
    def __init__(self, rate):
        self.rate = rate

    def get_rate(self):
        return self.rate