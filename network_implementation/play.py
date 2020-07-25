import pathlib, sys

path = pathlib.Path().absolute()
sys.path.insert(1, str(path))

from game.players import QNetPlayer, CmdPlayer
from game.gamestate import Game
from network_backend.Modules import ModuleI

net = ModuleI.fromFile("networks/q_learning/morris_ep_1200.net")
player0 = QNetPlayer(net, 0)
player1 = QNetPlayer(net, 1)
game = Game(p0=player0, p1=player1, run=False)
game.play(wait_and_show=True)
