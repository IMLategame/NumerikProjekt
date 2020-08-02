import pathlib, sys
import time
import glob
import os
import random
from copy import deepcopy
from pathos.multiprocessing import ProcessingPool as Pool

from matplotlib import pyplot as plt

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from TicTacToe.board import Board
from network_backend.Loss import L2Loss, CrossEntropyLoss
from network_backend.NonLinear import Tanh, ReLU, Sigmoid, Softmax, LeakyReLU
from network_backend.Optimizers import Adam, SGD
from TicTacToe.players import QNetPlayer, RandomPlayer, VNetPlayer, AlphaZeroPlayer
from network_backend.reinforcement_learning.utils import ReplayMem, DataMemory
from network_backend.reinforcement_learning.encodings import QEncoding, TTTQEncoding, TTTVEncoding
from TicTacToe.gamestate import Game
from network_backend.Modules import FullyConnectedNet, ModuleI, SequentialNetwork, LinearLayer
from network_backend.reinforcement_learning.rewardFunctions import SimpleReward, OnlyWinReward
from network_backend.reinforcement_learning.goalFunctions import QGoal, VGoal
from network_backend.Batching import SimpleBatcher, TwoGoalBatcher

folder = "networks/ttt_mcts_learning_v2/"
load_saved_version = False
offset = 0

net_pre = FullyConnectedNet([18, 20, 20, 20])
net_val = FullyConnectedNet([20, 1], nonLin=LeakyReLU())
net_distr = SequentialNetwork([FullyConnectedNet([20, 20], nonLin=LeakyReLU()), FullyConnectedNet([20, 9], nonLin=Softmax())])

if load_saved_version:
    assert offset != 0
    list_of_files = glob.glob(folder+"*.pre.net")
    latest_file = max(list_of_files, key=os.path.getctime)
    net_pre = ModuleI.fromFile(latest_file)
    list_of_files = glob.glob(folder + "*.dist.net")
    latest_file = max(list_of_files, key=os.path.getctime)
    net_distr = ModuleI.fromFile(latest_file)
    list_of_files = glob.glob(folder + "*.val.net")
    latest_file = max(list_of_files, key=os.path.getctime)
    net_val = ModuleI.fromFile(latest_file)
    print("used "+latest_file)
else:
    offset = 0

player0 = AlphaZeroPlayer(net_pre, net_val, net_distr, 0)
player1 = AlphaZeroPlayer(net_pre, net_val, net_distr, 1)
game = Game(run=False, p1=player0, p0=player1)

crit_d = CrossEntropyLoss()
crit_v = L2Loss()
memory = DataMemory(1000)

opt_pre = Adam(net_pre)
opt_val = Adam(net_val)
opt_distr = Adam(net_distr)

self_play_per_epoch = 100

workers = 10

epochs = 5000
evaluation_epochs = 100
evaluation_games = 100
save_epochs = 100
old_net_pre = None
old_net_val = None
old_net_distr = None

fig, ax = plt.subplots(1)
win_percentages = []


def simulate_games(g, N):
    game = deepcopy(g)
    game_data_list = []
    for _ in range(N):
        game.play()
        game_data_list.append(game.p0.get_data() + game.p1.get_data())
    return game_data_list


def test_games_p0_wins(g, N):
    game = deepcopy(g)
    wins = 0
    for _ in range(N):
        game.play()
        if game.winner is None:
            wins += 0.5
        elif game.winner == game.p0:
            wins += 1
    return wins


start = time.time()
for epoch in range(epochs+1):
    epoch += offset
    if epoch % save_epochs == 0:
        with open(folder + "ttt_ep_{}.pre.net".format(epoch), "w+") as f:
            net_pre.toFile(f)
        with open(folder + "ttt_ep_{}.dist.net".format(epoch), "w+") as f:
            net_distr.toFile(f)
        with open(folder + "ttt_ep_{}.val.net".format(epoch), "w+") as f:
            net_val.toFile(f)
    print("Epoch {}".format(epoch), end="\r", flush=True)
    assert evaluation_games % workers == 0
    N = int(evaluation_games / workers)
    with Pool(workers) as executor:
        results = executor.map(simulate_games, [game for _ in range(workers)], [N for _ in range(workers)])
        for result in results:
            for game_data in result:
                memory.add_game_data(game_data)
    """for _ in range(self_play_per_epoch):
        print("Epoch {} \t Game {}/{}".format(epoch, _+1, self_play_per_epoch), end="\r", flush=True)
        game.play()
        memory.add_game_data(player1.get_data() + player0.get_data())
    print("")"""
    batcher = TwoGoalBatcher(30, memory.get_data())
    for x, y_d, y_v in batcher:
        intermed = net_pre(x)
        out_d = net_distr(intermed)
        out_v = net_val(intermed)

        loss_d, delta_d = crit_d(out_d, y_d)
        loss_v, delta_v = crit_v(out_v, y_v)
        loss = loss_d + loss_v
        delta_inter_d = net_distr.backprop(delta_d)
        delta_inter_v = net_val.backprop(delta_v)
        delta_inter = delta_inter_d + delta_inter_v
        net_pre.backprop(delta_inter)

        opt_pre.take_step()
        opt_val.take_step()
        opt_distr.take_step()
    # Evaluation by self play vs older version:
    """if epoch % evaluation_epochs == 0:
        if old_net_pre is not None:
            # now fight
            player0.net_val = old_net_val
            player0.net_distr = old_net_distr
            player0.net_pre = old_net_pre
            wins = 0
            assert evaluation_games % workers == 0
            N = int(evaluation_games / workers)
            with Pool(workers) as executor:
                results = executor.map(test_games_p0_wins, [g for _ in range(workers)], [N for _ in range(workers)])
                for result in results:
                    wins += N - result
            g.p0 = player1
            g.p1 = player0
            with Pool(workers) as executor:
                results = executor.map(test_games_p0_wins, [g for _ in range(workers)], [N for _ in range(workers)])
                for result in results:
                    wins += result
            g.p0 = player0
            g.p1 = player1
            if wins/(2 * evaluation_games) >= 0.55:
                # use the new one
                player0.net_val = net_val
                player0.net_pre = net_pre
                player0.net_distr = net_distr
                print("Using the new net. Win percent was {}".format(wins/(2 * evaluation_games)))
            else:
                # keep the old one
                net_pre = old_net_pre
                net_val = old_net_val
                net_distr = old_net_distr
                player1.net_val = old_net_val
                player1.net_distr = old_net_distr
                player1.net_pre = old_net_pre
                print("Keeping the old net. Win percent was {}".format(wins/(2 * evaluation_games)))
        # save as the new standard
        old_net_pre = deepcopy(net_pre)
        old_net_distr = deepcopy(net_distr)
        old_net_val = deepcopy(net_val)"""
    # print("loss (last_epoch) = {:.6f} + {:.6f} = {:.6f}".format(sum(loss_d)/len(loss_d),
    #   sum(loss_v)/len(loss_v), sum(loss)/len(loss)), end='\r', flush=True)

    # Evaluation and plotting:
    if epoch % evaluation_epochs == 0:
        wins = 0
        print("Evaluating at Epoch {}: \t".format(epoch), end="\t", flush=True)
        g = Game(p0=player0, p1=RandomPlayer(1 - player0.playerID), run=False)
        assert evaluation_games % workers == 0
        N = int(evaluation_games / workers)
        with Pool(workers) as executor:
            results = executor.map(test_games_p0_wins, [g for _ in range(workers)], [N for _ in range(workers)])
            for result in results:
                wins += result
        g = Game(p0=RandomPlayer(1 - player1.playerID), p1=player1, run=False)
        with Pool(workers) as executor:
            results = executor.map(test_games_p0_wins, [g for _ in range(workers)], [N for _ in range(workers)])
            for result in results:
                wins += N - result
        win_percent = wins / (2 * evaluation_games)
        print("Won {:.2f} % of games".format(win_percent*100), end="\n", flush=True)
        win_percentages.append(win_percent)
        ax.clear()
        ax.plot(range(len(win_percentages)), win_percentages)
        ax.set_title("Win %")
        plt.pause(0.05)
        plt.draw()
end = time.time()
time_diff = end-start
print("Trained for \n\t {} s \n \t {} epochs\n \t {} s/epoch".format(time_diff, epoch+1, time_diff/(epoch+1)))

with open(folder + "ttt.pre.net".format(epoch), "w+") as f:
    net_pre.toFile(f)
with open(folder + "ttt.dist.net".format(epoch), "w+") as f:
    net_distr.toFile(f)
with open(folder + "ttt.val.net".format(epoch), "w+") as f:
    net_val.toFile(f)

plt.show()
