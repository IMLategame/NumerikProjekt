import glob
import os
import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from TicTacToe.players import QNetPlayer, CmdPlayer, VisualPlayer, MCTSPlayer, MiniMaxPlayer
from TicTacToe.gamestate import Game
from network_backend.Modules import ModuleI

player0 = MiniMaxPlayer(0)
player1 = VisualPlayer(1)
game = Game(p1=player0, p0=player1)
