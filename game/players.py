import numpy as np

from game.board import Board
from game.moves import Move
from utils import ReplayMem
import re

"""
    Interface for Players.
    Subclass this to create a new player (something that plays the game :) )
"""


class PlayerI:
    def __init__(self, playerID=0):
        # Player has an ID that is either 0 or 1
        if playerID in ["a", "b"]:
            playerID = {"a": 0, "b": 1}[playerID]
        assert playerID in [0, 1]
        self.playerID = playerID

    # This is the method to implement in 'real' player-classes. It takes a board and the game-phase and should in the implementation return a legal move.
    def get_move(self, phase, board: Board):
        assert phase in ["set", "move", "jump", "take"]
        raise NotImplementedError()

    # What the player does when it wins. Can change that in players.
    def win(self):
        print("Player {}: I won :)".format(self.playerID))


# Just parsing a string with some (>= 3) numbers in it to the numbers (only the first 3)
def parse_point(string):
    print("point stuff:")
    print([int(s) for s in re.findall(r'\d+', string)])
    r, x, y = [int(s) for s in re.findall(r'\d+', string)][:3]
    return r, x, y


"""
    The first and most simple implementation of a player. Just asks the user what to do on the command line.
"""


class CmdPlayer(PlayerI):
    def __init__(self, playerID=0):
        # Call this first in your implementations of the player also.
        super().__init__(playerID)

    # The function we have this class for
    def getMove(self, phase, board: Board):
        print("Its your turn. You are player {} \n The phase is {}".format(
            board.string_rep[board.player_map[self.playerID]], phase))
        print(board)
        # continue asking until we get a reasonable answer
        while True:
            print("Whats your move? Possibilities are 'mv <start>; <end>', 'st <point>' or 'tk <point>'.")
            move_string = input("->")
            # is the input possibly a command?
            if move_string[:2] not in ["mv", "st", "tk"]:
                continue

            # its a move command
            if move_string[:2] == "mv":
                # we have 2 points on the board with a ; somewhere in between!
                points = move_string[2:].split(";")
                start_p = parse_point(points[0])
                end_p = parse_point(points[1])
                move = Move("move", end_p, start_p)
            # its a set command
            elif move_string[:2] == "st":
                end_p = parse_point(move_string[2:])
                move = Move("set", end_p)
            # its a take command
            else:
                end_p = parse_point(move_string[2:])
                move = Move("take", end_p)
            # is this command even legal?
            if board.is_legal(move, phase, self.playerID):
                return move

class NetPlayerI(PlayerI):
    def __init__(self, net, playerID=0):
        super(NetPlayerI, self).__init__(playerID)

    def get_move(self, phase, board: Board):
        return super(NetPlayerI, self).get_move(phase, board)

    def win(self):
        super(NetPlayerI, self).win()

class QNetPlayer(NetPlayerI):
    def __init__(self, net, playerID=0):
        """
        :param net: R^127 -> R
        :param playerID: in [0,1]

        input vector: phases in R^4 x board in R^24*2 (my positions x enemy positions) x move.type in R^3 x move.start in R^24 x move.end in R^24

        board encoding (indices):

            2    ------------------------   11    ------------------------   17
            |                                |                                |
            |                                |                                |
            |                                |                                |
            |         1    --------------   10    --------------   16         |
            |         |                      |                      |         |
            |         |                      |                      |         |
            |         |                      |                      |         |
            |         |         0    ----    9    ----   15         |         |
            |         |         |                         |         |         |
            |         |         |                         |         |         |
            |         |         |                         |         |         |
            5    -    4    -    3                        18    -   19    -   20
            |         |         |                         |         |         |
            |         |         |                         |         |         |
            |         |         |                         |         |         |
            |         |         6    ----   12    ----   21         |         |
            |         |                      |                      |         |
            |         |                      |                      |         |
            |         |                      |                      |         |
            |         7    --------------   13    --------------   22         |
            |                                |                                |
            |                                |                                |
            |                                |                                |
            8    ------------------------   14    ------------------------   23

        """
        # Call this first in your implementations of the player also.
        super(QNetPlayer, self).__init__(playerID)
        self.net = net

    # encode all information one-hot.
    def encode(self, move, board, phase):
        assert phase in ["set", "move", "jump", "take"]
        phase2idx = {"set": 0, "move": 1, "jump": 2, "take": 3}
        moveType2idx = {"set": 0, "move": 1, "take": 2}
        phase_enc = [0.0, 0.0, 0.0, 0.0]
        phase_enc[phase2idx[phase]] = 1.0
        my_pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if board[r, x, y] == board.player_map[self.playerID]:
                        my_pos.append(1.0)
                    else:
                        my_pos.append(0.0)
        enemy_pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if board[r, x, y] == board.player_map[1-self.playerID]:
                        enemy_pos.append(1.0)
                    else:
                        enemy_pos.append(0.0)
        move_type_enc = [0.0, 0.0, 0.0]
        move_type_enc[moveType2idx[move.type]] = 1.0
        move_start = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if (r, x, y) == move.start:
                        move_start.append(1.0)
                    else:
                        move_start.append(0.0)
        move_end = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if (r, x, y) == move.end:
                        move_end.append(1.0)
                    else:
                        move_end.append(0.0)
        assert len(phase_enc) == 4
        assert len(my_pos) == 24
        assert len(enemy_pos) == 24
        assert len(move_type_enc) == 3
        assert len(move_start) == 24
        assert len(move_end) == 24
        enc = phase_enc + my_pos + enemy_pos + move_type_enc + move_start + move_end
        return np.array(enc)

    def get_move(self, phase, board: Board):
        legal_moves = board.legal_moves(phase, self.playerID)
        max_q = -2 ** 62
        max_action = None
        for move in legal_moves:
            encoded = self.encode(move, board, phase)
            q_val = self.net(encoded)
            if q_val > max_q:
                max_q = q_val
                max_action = move
        return move
