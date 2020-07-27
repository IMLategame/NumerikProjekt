import glob
import os
import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from TicTacToe.players import QNetPlayer, CmdPlayer
from TicTacToe.gamestate import Game
from network_backend.Modules import ModuleI

folder = "networks/ttt_q_learning/"

list_of_files = glob.glob(folder+"*.net")
latest_file = max(list_of_files, key=os.path.getctime)
print("using version "+latest_file)
net = ModuleI.fromFile(latest_file)
player0 = QNetPlayer(net, 0)
player1 = CmdPlayer(1)
game = Game(p0=player0, p1=player1, run=False)
game.play(wait_and_show=False)
