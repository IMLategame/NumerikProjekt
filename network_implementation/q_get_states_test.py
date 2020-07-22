from game.players import QNetPlayer
from utils import ReplayMem
from game.gamestate import Game
from network_backend.Modules import FullyConnectedNet

net = FullyConnectedNet([127, 50, 1])
player0 = QNetPlayer(net, 0)
player1 = QNetPlayer(net, 1)
memory = ReplayMem(100, 1, None, 1.0)
game = Game(run=False, p0=player0, p1=player1, mem=memory, reward=reward)
