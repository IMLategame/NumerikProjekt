from game.board import Board
from game.moves import Move
from game.players.players import PlayerI, CmdPlayer
import sys


class Game:
    def __init__(self):
        self.p0 = CmdPlayer(0)
        self.p1 = CmdPlayer(1)
        self.board = Board()
        self.no_pieces_set = 9
        self.play()

    def getMove(self, player: PlayerI, phase):
        while True:
            move = player.getMove(phase, self.board)
            if self.board.is_legal(move, phase, player.playerID):
                return move

    def play(self):
        # set phase
        for _ in range(self.no_pieces_set):
            move = self.getMove(self.p0, "set")
            self.board.do(move, self.p0.playerID)
            move = self.getMove(self.p1, "set")
            self.board.do(move, self.p1.playerID)
        # move phase
        while len(self.board.get_player_pos(self.p0.playerID)) > 2 and len(self.board.get_player_pos(self.p1.playerID)) > 2:
            if len(self.board.get_player_pos(self.p0.playerID)) == 3:
                move = self.getMove(self.p0, "jump")
            else:
                move = self.getMove(self.p0, "move")
            self.board.do(move, self.p0.playerID)
            if self.board.in_mull(self.p0.playerID, move.end):
                move = self.getMove(self.p0, "take")
                self.board.do(move, self.p0.playerID)
            if len(self.board.get_player_pos(self.p1.playerID)) == 3:
                move = self.getMove(self.p1, "jump")
            else:
                move = self.getMove(self.p1, "move")
            self.board.do(move, self.p1.playerID)
            if self.board.in_mull(self.p1.playerID, move.end):
                move = self.getMove(self.p1, "take")
                self.board.do(move, self.p1.playerID)
        # check winner
        if len(self.board.get_player_pos(self.p0.playerID)) <= 2:
            self.p0.win()
        else:
            self.p1.win()
