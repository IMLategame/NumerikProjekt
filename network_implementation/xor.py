import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from network_backend.Modules import ModuleI, FullyConnectedNet, SequentialNetwork, LinearLayer, NonLinearLayer
from network_backend.Loss import BCELoss, L2Loss
from network_backend.Optimizers import SGD, Adam
from network_backend.NonLinear import ReLU, Sigmoid, Identity, Tanh
import numpy as np
import random, time

data = [[np.array((0, 0)), np.array(0)], [np.array((0, 1)), np.array(1)], [np.array((1, 0)), np.array(1)], [np.array((1, 1)), np.array(0)]]

net = FullyConnectedNet([2, 3, 1], nonLin=Tanh())
#net = SequentialNetwork([FullyConnectedLayer(2, 3, Identity()), NonLinearLayer(Sigmoid()), FullyConnectedLayer(3, 1, Sigmoid())])
criterion = L2Loss() # BCELoss()  #
opt = Adam(net)  # SGD(net, 0.001) #
epochs = 500000
eval = 5000
start = time.time()
for epoch in range(epochs):
    random.shuffle(data)
    for x, y in data:
        out = net(x)
        loss, delta = criterion(out, y)
        net.backprop(delta)
        opt.take_step()
    loss = sum([criterion(net(x), y)[0] for x, y in data])
    if loss < 0.1:
        break
    if (epoch+1) % eval == 0:
        for x, y in data:
            print("{0[0]}, \t{0[1]} \t-> \t{1[0]:.6f} \t vs \t{2}".format(x, net(x)[0], y))
        print("loss = {}".format(loss))
end = time.time()
time_diff = end-start

print("Trained for \n\t {} s \n \t {} epochs\n \t {} s/epoch".format(time_diff, epoch+1, time_diff/(epoch+1)))

print("Final Values:")
for x, y in data:
    print("{0[0]}, \t{0[1]} \t-> \t{1[0]:.6f} \t vs \t{2}".format(x, net(x)[0], y))
loss = sum([criterion(net(x), y)[0] for x, y in data])
print("final loss = {0[0]} ".format(loss))

# tests for store and load:
"""
print(net.toDict())
print(net)
string = str(net)
net2 = ModuleI.fromString(string)
print(net2.toDict())

for x, y in data:
    print("{0[0]}, \t{0[1]} \t-> \t{1[0]:.6f} \t vs \t{2}".format(x, net(x), y))
    print("{0[0]}, \t{0[1]} \t-> \t{1[0]:.6f} \t vs \t{2}".format(x, net2(x), y)) """
