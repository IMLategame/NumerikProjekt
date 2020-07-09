import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from network_backend.Module import ModuleI
from network_backend.Loss import BCELoss, L2Loss
from network_backend.Networks import FullyConnectedNet
from network_backend.Optimizers import SGD, Adam
import numpy as np
import random

data = [[np.array((0, 0)), np.array(0)], [np.array((0, 1)), np.array(1)], [np.array((1, 0)), np.array(1)], [np.array((1, 1)), np.array(0)]]

net = FullyConnectedNet([2, 3, 1])
criterion = BCELoss()  # L2Loss()
opt = Adam(net)  # SGD(net, 0.001)
epochs = 50000
eval = 5000
for epoch in range(epochs):
    random.shuffle(data)
    for x, y in data:
        out = net(x)
        loss, delta = criterion(out, y)
        net.backprop(delta)
        opt.take_step()
    loss = sum([criterion(net(x), y)[0] for x, y in data])
    if (epoch+1) % eval == 0:
        print("{} \t last loss: {}".format(epoch+1, loss))
        for x, y in data:
            print("{0[0]}, \t{0[1]} \t-> \t{1[0]:.6f} \t vs \t{2}".format(x, net(x), y))
        print("loss = {}".format(loss))

print("Final Values:")
for x, y in data:
    print("{0[0]}, \t{0[1]} \t-> \t{1[0]:.6f} \t vs \t{2}".format(x, net(x), y))
loss = sum([criterion(net(x), y)[0] for x,y in data])
print("final loss = {0[0]} ".format(loss))