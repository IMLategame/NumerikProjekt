import numpy as np
from network_backend.NonLinear import Sigmoid, ActivationI, fctn_dict, Identity
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

    def feed_forward(self, x):
        """
            :param x: input: shape = (input size, batch size)
            :return: output: shape = (output size, batch size)
        """
        raise NotImplementedError()

    def backprop(self, delta_out):
        """
            Do backpropagatiion for the last input and save the gradients.
            :param delta_out: change in output
            :return: delta_in: corresponding change in input
        """
        raise NotImplementedError()

    def update(self, delta):
        """
            parameters -= delta
            :param delta: differences
        """
        raise NotImplementedError()

    def noFeatures(self):
        """
            :return: number of elements of the gradient list: len(self.getGradients())
        """
        raise NotImplementedError()

    def set_mode(self, mode):
        """
            Set the current mode of evaluation. Needed for Dropout, etc.
            :param mode: train or test
        """
        if mode in ['train', 'Train']:
            self.mode_train = True
        elif mode in ['test', 'Test', 'eval', 'Eval']:
            self.mode_train = False

    def getGradients(self):
        """
            :return: Gradients for the last run of backpropagation.
        """
        raise NotImplementedError()

    def toDict(self):
        """
            :return: The module as a python dictionary. For saving and restoring.
        """
        obj = {"class_name": type(self).__name__}
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        """
            Resore the module from a python dictionary.
            :param obj: dictionary
            :return: module
        """
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


def FullyConnectedNet(sizes, nonLin=Sigmoid()):
    layers = []
    assert len(sizes) >= 2
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        layers.append(LinearLayer(size_in, size_out))
        layers.append(NonLinearLayer(nonLin))
    return SequentialNetwork(layers)


class LinearLayer(ModuleI):
    def __init__(self, lay_in, lay_out):
        super(LinearLayer, self).__init__()
        self.weights = np.random.normal(loc=0.0, scale=0.3, size=(lay_out, lay_in))
        self.bias = np.zeros(lay_out)
        self.lay_in = lay_in
        self.lay_out = lay_out

    def feed_forward(self, x):
        if len(x.shape) <= 1:
            x = x.reshape((x.shape[0], 1))
        self.x = x
        z = self.weights @ x + self.bias[:, np.newaxis]
        return z

    def backprop(self, delta_out):
        delta_in = self.weights.T @ delta_out
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
        obj = super(LinearLayer, self).toDict()
        obj["lay_in"] = self.lay_in
        obj["lay_out"] = self.lay_out
        obj["weights"] = self.weights.tolist()
        obj["bias"] = self.bias.tolist()
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        layer = LinearLayer(obj["lay_in"], obj["lay_out"])
        layer.weights = np.array(obj["weights"])
        layer.bias = np.array(obj["bias"])
        return layer


class NonLinearLayer(ModuleI):
    def __init__(self, non_linearity):
        super().__init__()
        assert isinstance(non_linearity, ActivationI)
        self.nonLin = non_linearity

    def feed_forward(self, x):
        self.y = self.nonLin(x)
        return self.y

    def backprop(self, delta_out):
        return self.nonLin.d(self.y) * delta_out

    def noFeatures(self):
        return 0

    def update(self, delta):
        pass

    def getGradients(self):
        return []

    def toDict(self):
        obj = super(NonLinearLayer, self).toDict()
        obj["non_linearity"] = self.nonLin.toDict()
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        return NonLinearLayer(ActivationI.fromDict(obj["non_linearity"]))


"""class ResidualLayer(ModuleI):
    def __init__(self, subnet: ModuleI):
        super(ResidualLayer, self).__init__()
        self.net = subnet

    def feed_forward(self, x):
        return x + self.net(x)

    def backprop(self, delta_out):
        return self.net.backprop(delta_out) + delta_out

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
        return ResidualLayer(ModuleI.fromDict(obj["subnet"]))"""


class SplitNonLinearLayer(ModuleI):
    def __init__(self, sizes, non_linearities):
        super(SplitNonLinearLayer, self).__init__()
        self.sizes = sizes
        self.non_linearities = non_linearities
        self.split_sizes = [sum(self.sizes[:i+1]) for i in range(len(self.sizes))]
        assert len(sizes) >= 1
        assert len(sizes) == len(non_linearities)
        for non_lin in non_linearities:
            assert isinstance(non_lin, ActivationI)

    def feed_forward(self, x):
        assert x.shape[0] == sum(self.sizes)
        split_x = np.split(x, self.split_sizes, axis=0)
        split_x = [non_lin(x_i) for x_i, non_lin in zip(split_x, self.non_linearities)]
        return np.concatenate(split_x, axis=0)

    def backprop(self, delta_out):
        assert delta_out.shape[0] == sum(self.sizes)
        delta_split = np.split(delta_out, self.split_sizes, axis=0)
        delta_split = [non_lin.d(delta_i) for delta_i, non_lin in zip(delta_split, self.non_linearities)]
        return np.concatenate(delta_split, axis=0)

    def noFeatures(self):
        return 0

    def update(self, delta):
        pass

    def getGradients(self):
        return []

    def toDict(self):
        obj = super(SplitNonLinearLayer, self).toDict()
        obj["sizes"] = self.sizes
        obj["non_linearities"] = [non_lin.toDict() for non_lin in self.non_linearities]
        return obj

    @classmethod
    def dict2Mod(cls, obj):
        sizes = obj["sizes"]
        non_lins = [ActivationI.fromDict(non_lin) for non_lin in obj["non_linearities"]]
        return SplitNonLinearLayer(sizes, non_lins)


class_dict = {
    "LinearLayer": LinearLayer,
    "SequentialNetwork": SequentialNetwork,
    "NonLinearLayer": NonLinearLayer,
    #"ResidualLayer": ResidualLayer,
    "SplitNonLinearLayer": SplitNonLinearLayer
}
