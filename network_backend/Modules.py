import numpy as np
from network_backend.NonLinear import Sigmoid, NonLinearI, fctn_dict
import json


class ModuleI:
    # call super().__init__() in subclasses
    def __init__(self):
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

    @classmethod
    def fromFile(cls, path):
        with open(path) as f:
            for l in f:
                return cls.fromString(l)

    @classmethod
    def fromString(cls, str):
        return cls.fromDict(json.loads(str))

    @classmethod
    def fromDict(cls, obj):
        return class_dict[obj["class_name"]].dict2Mod(obj)

    def toString(self):
        return json.dumps(self.toDict())

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

    def toDict(self):
        obj = {"class_name": type(self).__name__}
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        raise NotImplementedError()


class SequentialNetwork(ModuleI):
    def __init__(self, layers):
        super(SequentialNetwork, self).__init__()
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

    def toDict(self):
        obj = super(SequentialNetwork, self).toDict()
        layer_list = [layer.toDict() for layer in self.layers]
        obj["layers"] = layer_list
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        layer_list = obj["layers"]
        layers = [ModuleI.fromDict(lay) for lay in layer_list]
        return SequentialNetwork(layers)


def FullyConnectedNet(sizes = None, nonLin = Sigmoid()):
    layers = []
    assert len(sizes) >= 2
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        layers.append(FullyConnectedLayer(size_in, size_out, nonLin))
    return SequentialNetwork(layers)


class FullyConnectedLayer(ModuleI):
    def __init__(self, lay_in, lay_out, fctn=Sigmoid()):
        super(FullyConnectedLayer, self).__init__()
        assert isinstance(fctn, NonLinearI)
        self.weights = np.random.normal(size=(lay_out, lay_in))
        self.bias = np.zeros(lay_out)
        self.fctn = fctn
        self.lay_in = lay_in
        self.lay_out = lay_out

    def feed_forward(self, x):
        if len(x.shape) <= 1:
            x = x.reshape((x.shape[0], 1))
        self.x = x
        z = self.weights @ x + self.bias[:, np.newaxis]
        return self.fctn(z)

    def backprop(self, delta_out):
        delta_in = self.fctn.d(self.x) * (self.weights.T @ delta_out)
        self.der_b = delta_out
        self.der_w = delta_out @ self.x.T
        return delta_in

    def noFeatures(self):
        return 2

    def update(self, delta):
        self.weights -= delta[0]
        self.bias -= delta[1]

    def getGradients(self):
        if len(self.der_b.shape) > 1:
            n = self.der_w.shape[-1]
            self.der_b = np.sum(self.der_b, axis=-1)/n
        return [self.der_w, self.der_b]

    def toDict(self):
        obj = super(FullyConnectedLayer, self).toDict()
        obj["lay_in"] = self.lay_in
        obj["lay_out"] = self.lay_out
        obj["weights"] = self.weights.tolist()
        obj["bias"] = self.bias.tolist()
        obj["non_linearity"] = type(self.fctn).__name__
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        layer = FullyConnectedLayer(obj["lay_in"], obj["lay_out"], fctn=fctn_dict[obj["non_linearity"]]())
        layer.weights = np.array(obj["weights"])
        layer.bias = np.array(obj["bias"])
        return layer


class NonLinearLayer(ModuleI):
    def __init__(self, non_linearity):
        super().__init__()
        assert isinstance(non_linearity, NonLinearI)
        self.nonLin = non_linearity

    def feed_forward(self, x):
        return self.nonLin(x)

    def backprop(self, delta_out):
        return self.nonLin.d(delta_out)

    def noFeatures(self):
        return 0

    def update(self, delta):
        pass

    def getGradients(self):
        return []

    def toDict(self):
        obj = super(NonLinearLayer, self).toDict()
        obj["non_linearity"] = type(self.nonLin).__name__
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        return NonLinearLayer(fctn_dict[obj["non_linearity"]])


class ResidualLayer(ModuleI):
    def __init__(self, subnet: ModuleI):
        super(ResidualLayer, self).__init__()
        self.net = subnet

    def feed_forward(self, x):
        return x + self.net(x)

    def backprop(self, delta_out):
        return delta_out + self.net.backprop(delta_out)

    def noFeatures(self):
        return self.net.noFeatures()

    def update(self, delta):
        self.net.update(delta)

    def getGradients(self):
        return self.net.getGradients()

    def toDict(self):
        obj = super(ResidualLayer, self).toDict()
        obj["subnet"] = self.net.toDict()
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        return ResidualLayer(ModuleI.fromDict(obj["subnet"]))


class_dict = {
    "FullyConnectedLayer": FullyConnectedLayer,
    "SequentialNetwork": SequentialNetwork,
    "NonLinearLayer": NonLinearLayer,
    "ResidualLayer": ResidualLayer
}
