from game.board import Board
from game.moves import Move
from game.players.players import PlayerI, CmdPlayer
import sys

class Game:
    def __init__(self):
        self.p0 = CmdPlayer(0)
        self.p1 = CmdPlayer(1)
        self.board = Board()
        self.phase = "set"
        self.play()

    def getMove(self, player: PlayerI):
        while True:
            move = player.getMove(self.phase, self.board)
            if self.board.is_legal(move, self.phase, player.playerID):
                return move

    def play(self):
        self.phase = "set"
        for _ in range()
