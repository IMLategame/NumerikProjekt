import glob
import os
import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from TicTacToe.players import QNetPlayer, CmdPlayer, VisualPlayer, MCTSPlayer, AlphaZeroPlayer
from TicTacToe.gamestate import Game
from network_backend.Modules import ModuleI

folder = "networks/ttt_mcts_learning_enemy_gets_neg_val/"

list_of_files = glob.glob(folder+"*.pre.net")
latest_file = "networks/ttt_alphaZero_final.pre.net"
print("used " + latest_file)
net_pre = ModuleI.fromFile(latest_file)
list_of_files = glob.glob(folder + "*.dist.net")
latest_file = "networks/ttt_alphaZero_final.dist.net"
print("used " + latest_file)
net_distr = ModuleI.fromFile(latest_file)
list_of_files = glob.glob(folder + "*.val.net")
latest_file = "networks/ttt_alphaZero_final.val.net"
net_val = ModuleI.fromFile(latest_file)
print("used "+latest_file)

player0 = AlphaZeroPlayer(net_pre, net_val, net_distr, 0)
player1 = VisualPlayer(1)
game = Game(p1=player0, p0=player1, run=False)
game.play(wait_and_show=False)
