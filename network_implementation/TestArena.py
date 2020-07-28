import pathlib, sys
import time
import glob
import os
import random
from copy import deepcopy

from matplotlib import pyplot as plt

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from TicTacToe.board import Board
from network_backend.Loss import L2Loss
from network_backend.NonLinear import Tanh, ReLU, Sigmoid
from network_backend.Optimizers import Adam, SGD
from TicTacToe.players import QNetPlayer, RandomPlayer, VNetPlayer, MinMaxPlayer, VisualPlayer
from network_backend.reinforcement_learning.utils import ReplayMem
from network_backend.reinforcement_learning.encodings import QEncoding, TTTQEncoding, TTTVEncoding
from TicTacToe.gamestate import Game
from network_backend.Modules import FullyConnectedNet, ModuleI, SequentialNetwork, LinearLayer
from network_backend.reinforcement_learning.rewardFunctions import SimpleReward, OnlyWinReward
from network_backend.reinforcement_learning.goalFunctions import QGoal, VGoal
from network_backend.Batching import SimpleBatcher

folder = "networks/ttt_v_learning/"

list_of_files = glob.glob(folder+"*.net")
latest_file = max(list_of_files, key=os.path.getctime)
print("using version "+latest_file)
net = ModuleI.fromFile(latest_file)
player0 = VNetPlayer(net, 0)
player1 = VNetPlayer(net, 1)
start = time.time()
game = Game(p0=player0, p1=player1, run=True)
print("First game winner is player {}".format(game.winner))
game = Game(p1=player0, p0=player1, run=True)
print("Second game winner is player {}".format(game.winner))
end = time.time()
print("played for {} s".format(end-start))
