from network_backend.Layers import FullyConnectedLayer
from network_backend.Module import ModuleI
import json

from network_backend.NonLinear import Sigmoid


class SequentialNetwork(ModuleI):
    def __init__(self, layers=None, str=None):
        super(SequentialNetwork, self).__init__(str)
        if str is not None:
            return
        for layer in layers:
            assert isinstance(layer, ModuleI)
        self.layers = layers

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backprop(self, delta_out):
        delta = delta_out
        for layer in reversed(self.layers):
            delta = layer.backprop(delta)
        return delta

    def noFeatures(self):
        return sum([len(layer) for layer in self.layers])

    def getGradients(self):
        return [grad for layer in self.layers for grad in layer.getGradients()]

    def update(self, delta):
        for layer in self.layers:
            layer.update(delta[:len(layer)])
            delta = delta[len(layer):]

    def toString(self):
        string = "["
        for l in self.layers:
            string += str(l) + ", "
        string = string[:-1]+ "]"
        return string

def FullyConnectedNet(sizes = None, nonLin = Sigmoid(), str=None):
    layers = []
    if str is not None:
        return SequentialNetwork(str=str)
    assert len(sizes) >= 2
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        layers.append(FullyConnectedLayer(size_in, size_out, nonLin))
    return SequentialNetwork(layers)
