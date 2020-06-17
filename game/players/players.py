from game.board import Board
from game.moves import Move
import re

"""Interface for Players"""


class PlayerI:
    def __init__(self, playerID=0):
        assert playerID in [0, 1]
        self.playerID = playerID

    def set_playerID(self, ID):
        assert ID in [0, 1]
        self.playerID = ID

    def get_move(self, phase, board: Board):
        assert phase in ["set", "move", "jump", "take"]
        raise NotImplementedError()


def parse_point(string):
    r, x, y = [int(s) for s in re.findall(r'\d+', string)]
    return r, x, y


class CmdPlayer(PlayerI):
    def __init__(self, playerID=0):
        super(self, CmdPlayer).__init__(playerID)

    def getMove(self, phase, board: Board):
        print("Its your turn. You are player {} \n The phase is {}".format(board.string_rep[self.playerID], phase))
        print(board)
        while True:
            print("Whats your move? Possibilities are 'mv <start>; <end>', 'st <point>' or 'tk <point>'.")
            move_string = input("->")
            if move_string[:2] not in ["mv", "st", "tk"]:
                continue
            if move_string[:2] == "mv":
                points = move_string[2:].split("; ")
                start_p = parse_point(points[0])
                end_p = parse_point(points[1])
                move = Move("move", end_p, start_p)
            elif move_string[:2] == "st":
                end_p = parse_point(move_string[2:])
                move = Move("set", end_p)
            else:
                end_p = parse_point(move_string[2:])
                move = Move("take", end_p)
            if board.is_legal(move, phase, self.playerID):
                return move
