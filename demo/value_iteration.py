import glob
import os
import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from TicTacToe.players import QNetPlayer, CmdPlayer, VisualPlayer, MCTSPlayer, VNetPlayer
from TicTacToe.gamestate import Game
from network_backend.Modules import ModuleI

file = "networks/ttt_v_final.net"

print("using version "+file)
net = ModuleI.fromFile(file)
player0 = VNetPlayer(net, 0)
player1 = VisualPlayer(1)
game = Game(p0=player0, p1=player1)
