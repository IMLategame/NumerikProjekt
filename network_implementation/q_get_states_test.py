import pathlib, sys
import time
from matplotlib import pyplot as plt

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from network_backend.Loss import L2Loss
from network_backend.NonLinear import Tanh
from network_backend.Optimizers import Adam
from game.players import QNetPlayer, RandomPlayer
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
memory = ReplayMem(1000, 1, net, gamma=0.99, encode=QEncoding(), goal_value_function=QGoal())
reward = SimpleReward(take_factor=10.0, penalty=0.01, win_reward=100.0)
game = Game(run=False, p0=player0, p1=player1, mem=memory, reward=reward)

opt = Adam(net)
criterion = L2Loss()
batch_size = 30
epochs = 1000
plt.axis([0, epochs, 0, 10])
start = time.time()
losses = []
evaluation_epochs = 10
evaluation_games = 10
assert epochs % evaluation_epochs == 0
plt.axis([0, int(epochs/evaluation_epochs), 0, 1])
for epoch in range(epochs):
    if epoch % epochs == 0:
        with open("networks/q_learning/morris_ep_{}.net".format(epoch), "w+") as f:
            net.toFile(f)
    print("Epoch {}".format(epoch))
    n = game.learn(0.1)
    print("\tPlayed for {} steps".format(n))
    batcher = SimpleBatcher(batch_size, memory.get_data())
    for x, y in batcher:
        out = net(x)
        loss, delta = criterion(out, y)
        net.backprop(delta)
        opt.take_step()
    # Evaluation and plotting:
    if epoch % evaluation_epochs == 0:
        print("Evaluating")
        wins = 0
        g = Game(p0=player0, p1=RandomPlayer(1 - player0.playerID), run=False)
        for _ in range(evaluation_games):
            g.play()
            if g.winner.playerID == player0.playerID:
                wins += 1
        g = Game(p0=RandomPlayer(1 - player1.playerID), p1=player1, run=False)
        for _ in range(evaluation_games):
            g.play()
            if g.winner.playerID == player1.playerID:
                wins += 1
        win_percent = wins/(2 * evaluation_games)
        print("\tWon {}% of games".format(win_percent * 100))
        plt.scatter(int(epoch/evaluation_epochs), win_percent)
        plt.pause(0.05)
end = time.time()
time_diff = end-start
print("Trained for \n\t {} s \n \t {} epochs\n \t {} s/epoch".format(time_diff, epoch+1, time_diff/(epoch+1)))

with open("networks/q_learning/morris.net", "w+") as f:
    net.toFile(f)
