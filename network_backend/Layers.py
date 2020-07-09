from network_backend.Module import ModuleI
from network_backend.NonLinear import NonLinearI, Sigmoid
import numpy as np
import json

class FullyConnectedLayer(ModuleI):
    def __init__(self, lay_in=None, lay_out=None, fctn=Sigmoid(), str=None):
        super(FullyConnectedLayer, self).__init__(str)
        if str is not None:
            return
        assert lay_in is not None and lay_out is not None
        assert isinstance(fctn, NonLinearI)
        self.weights = np.random.normal(size=(lay_out, lay_in))
        self.bias = np.random.normal(size=lay_out)
        self.fctn = fctn
        self.lay_in = lay_in
        self.lay_out = lay_out

    def feed_forward(self, x):
        self.x = x
        self.z = self.weights @ x + self.bias
        return self.fctn(self.z)

    def backprop(self, delta_out):
        delta_in = self.fctn.d(self.x) * (self.weights.T @ delta_out)
        self.der_b = delta_out
        self.der_w = np.outer(delta_out, self.x)
        return delta_in

    def noFeatures(self):
        return 2

    def update(self, delta):
        self.weights -= delta[0]
        self.bias -= delta[1]

    def getGradients(self):
        return [self.der_w, self.der_b]

    def toString(self):
        obj = {"lay_in": self.lay_in, "lay_out": self.lay_out, "weights": str(self.weights.tostring()), "bias": str(self.bias.tostring())}
        return str(json.dumps(obj))