# This is just so that you can run the NineMenMorris more easily. Nothing to worry about here.
import sys
import pathlib
path = pathlib.Path().absolute()
sys.path.insert(1, str(path))
from NineMenMorris.gamestate import Game

g = Game()