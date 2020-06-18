import sys
import pathlib
path = pathlib.Path().absolute()
sys.path.insert(1, str(path))
from game.gamestate import Game

g = Game()