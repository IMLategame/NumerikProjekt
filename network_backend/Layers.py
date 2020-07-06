from network_backend.Module import ModuleI
from network_backend.NonLinear import NonLinearI
import numpy as np

class SingleLayer(ModuleI):
    def __init__(self, lay_in, lay_out, fctn):
        super(SingleLayer, self).__init__()
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
        delta_in = self.fctn.d(self.z) * (self.weights.T @ delta_out)
        self.der_b = delta_out
        self.der_w = delta_out @ self.x.T
        return delta_in

    def noFeatures(self):
        return 2

    def update(self, delta):
        self.weights -= delta[0]
        self.bias -= delta[1]

    def getGradients(self):
        return [self.der_w, self.der_b]

    def toString(self):
        string = "{" + str(self.lay_in) + "; " + str(self.lay_out) + "; " + self.weights.tostring() + "; " + self.bias.tostring() + "}"
        return string

    def fromString(self, string):
        assert string[0] == '{'
        assert string[-1] == '}'
        string_list = string[1:-1].split("; ")
        self.lay_in = int(string_list[0])
        self.lay_out = int(string_list[1])
        self.weights = np.fromstring(string_list[2]).reshape((self.lay_out, self.lay_in))
        self.bias = np.fromstring(string_list[3])