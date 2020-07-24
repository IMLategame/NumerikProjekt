import pathlib, sys
import time
from matplotlib import pyplot as plt

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from network_backend.Loss import L2Loss
from network_backend.NonLinear import Tanh
from network_backend.Optimizers import Adam
from game.players import QNetPlayer
from network_backend.reinforcement_learning.utils import ReplayMem
from network_backend.reinforcement_learning.encodings import QEncoding
from game.gamestate import Game
from network_backend.Modules import FullyConnectedNet, ModuleI
from network_backend.reinforcement_learning.rewardFunctions import SimpleReward
from network_backend.reinforcement_learning.goalFunctions import QGoal
from network_backend.Batching import SimpleBatcher

net = FullyConnectedNet([103, 100, 50, 200, 50, 10, 1], nonLin=Tanh())
#net = ModuleI.fromFile("networks/q_learning/morris_ep_200.net")

player0 = QNetPlayer(net, 0)
player1 = QNetPlayer(net, 1)
memory = ReplayMem(1000, 1, net, 0.9, encode=QEncoding(), goal_value_function=QGoal())
reward = SimpleReward(take_factor=1.0, penalty=0.1, win_reward=10.0)
game = Game(run=False, p0=player0, p1=player1, mem=memory, reward=reward)

opt = Adam(net)
criterion = L2Loss()
batch_size = 30
epochs = 100
epoch = 200
#plt.axis([0, epochs+epoch, 0, 10])
start = time.time()
losses = []
plt.axis([0, epochs, 0, 10])
while True:
    if epoch % epochs == 0:
        with open("networks/q_learning/morris_ep_{}.net".format(epoch), "w+") as f:
            net.toFile(f)
    n = game.learn(0.1)
    print("Epoch {}\n\tPlayed for {} steps".format(epoch, n))
    batcher = SimpleBatcher(batch_size, memory.get_data())
    for x, y in batcher:
        out = net(x)
        loss, delta = criterion(out, y)
        net.backprop(delta)
        opt.take_step()
    batch_avrg_loss = sum(loss)/batch_size
    losses.append(batch_avrg_loss)
    print(batch_avrg_loss)
    #plt.scatter(epoch, batch_avrg_loss)
    #plt.pause(0.05)
    epoch += 1
end = time.time()
time_diff = end-start
print("Trained for \n\t {} s \n \t {} epochs\n \t {} s/epoch".format(time_diff, epoch+1, time_diff/(epoch+1)))

with open("networks/q_learning/morris.net", "w+") as f:
    net.toFile(f)
