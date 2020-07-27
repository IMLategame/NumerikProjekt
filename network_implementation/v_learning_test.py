import pathlib, sys
import time
import glob
import os
import random
from matplotlib import pyplot as plt

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from network_backend.Loss import L2Loss
from network_backend.NonLinear import Tanh
from network_backend.Optimizers import Adam, SGD
from game.players import VNetPlayer, RandomPlayer
from network_backend.reinforcement_learning.utils import ReplayMem
from network_backend.reinforcement_learning.encodings import VEncoding
from game.gamestate import Game
from network_backend.Modules import FullyConnectedNet, ModuleI, SequentialNetwork, LinearLayer
from network_backend.reinforcement_learning.rewardFunctions import SimpleReward
from network_backend.reinforcement_learning.goalFunctions import VGoal
from network_backend.Batching import SimpleBatcher

net = SequentialNetwork([FullyConnectedNet([52, 100, 50, 50, 10]),
                         LinearLayer(10, 1)])
folder = "networks/v_learning/"
load_saved_version = False
offset = 0

if load_saved_version:
    list_of_files = glob.glob(folder+"*.net")
    latest_file = max(list_of_files, key=os.path.getctime)
    net = ModuleI.fromFile(latest_file)

player0 = VNetPlayer(net, 0)
player1 = VNetPlayer(net, 1)
memory = ReplayMem(2000, 1, net, gamma=0.9, encode=VEncoding(), goal_value_function=VGoal())
reward = SimpleReward(take_factor=10.0, penalty=0.01, win_reward=100.0)
game = Game(run=False, p0=player0, p1=player1, mem=memory, reward=reward)

opt = Adam(net)
criterion = L2Loss()


batch_size = 30
epochs = 10000
plt.axis([0, epochs, 0, 10])
start = time.time()
save_epochs = 100
evaluation_epochs = 25
evaluation_games = 10
assert epochs % evaluation_epochs == 0
eval_set_size = 50
max_learning_epochs_per_play_epoch = 10

fig, ax = plt.subplots(3)
win_percentages = []
losses_on_fixed_states = []
avrg_q_fixed_states = []

# collect set of states for observation of loss and q on fixed states:
game.learn(0.1)
test_data = memory.get_current_mem()
random.shuffle(test_data)
test_data = test_data[: min([len(test_data), eval_set_size])]
eval_mem = ReplayMem(2000, 1, net, gamma=0.99, encode=VEncoding(), goal_value_function=VGoal())
eval_mem.mem = test_data
print("Got {} set states for evaluation".format(len(test_data)))

for epoch in range(epochs):
    epoch += offset
    if epoch % save_epochs == 0:
        with open(folder + "morris_ep_{}.net".format(epoch), "w+") as f:
            net.toFile(f)
    print("Epoch {}".format(epoch))
    n = game.learn(0.1)
    print("\tPlayed for {} steps".format(n))
    batcher = SimpleBatcher(batch_size, memory.get_data())
    test_batcher = batcher.subset_percent(0.1)

    # Learn until no improvements are made
    i = 0
    learning = True
    last_loss = float("inf")
    while learning:
        for x, y in batcher:
            out = net(x)
            loss, delta = criterion(out, y)
            net.backprop(delta)
            opt.take_step()

        test_loss = 0
        for x, y in test_batcher:
            loss, _ = criterion(net(x), y)
            test_loss += sum(loss)/len(loss)
        test_loss = test_loss/len(test_batcher)
        if test_loss > last_loss or i >= max_learning_epochs_per_play_epoch-1:
            learning = False
        last_loss = test_loss
        i += 1
    print("\tLearned for {} epochs".format(i))
    print("\tFinal test loss = {}".format(test_loss))

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
        avrg_q_batcher = SimpleBatcher(10, eval_mem.get_data())
        win_percentages.append(win_percent)
        ax[0].clear()
        ax[0].plot(range(len(win_percentages)), win_percentages)
        ax[0].set_title("Win %")

        batch_avrg_loss = 0
        avrg_q = 0
        for x, y in avrg_q_batcher:
            q = net(x)
            loss, _ = criterion(q, y)
            batch_avrg_loss += sum(loss)/len(loss)
            avrg_q += sum(q)/len(q)
        batch_avrg_loss = batch_avrg_loss/len(avrg_q_batcher)
        print("\tAvrg loss = {}".format(batch_avrg_loss))
        losses_on_fixed_states.append(batch_avrg_loss)
        avrg_q = sum(avrg_q)/(len(avrg_q_batcher)*len(avrg_q))
        print("\tAvrg q = {}".format(avrg_q))
        avrg_q_fixed_states.append(avrg_q)
        ax[1].clear()
        ax[1].plot(range(len(losses_on_fixed_states)), losses_on_fixed_states)
        ax[1].set_title("Avrg loss on fixed States")
        ax[2].clear()
        ax[2].plot(range(len(avrg_q_fixed_states)), avrg_q_fixed_states)
        ax[2].set_title("Avrg Q on fixed states")
        plt.pause(0.1)
        plt.draw()
end = time.time()
time_diff = end-start
print("Trained for \n\t {} s \n \t {} epochs\n \t {} s/epoch".format(time_diff, epoch+1, time_diff/(epoch+1)))

with open(folder + "morris.net", "w+") as f:
    net.toFile(f)

plt.show()
