from NineMenMorris.board import Board
from NineMenMorris.moves import Move
from network_backend.reinforcement_learning.encodings import QEncoding, VEncoding
import re
from random import random, sample
from copy import deepcopy


class PlayerI:
    """
        Interface for Players.
        Subclass this to create a new player (something that plays the NineMenMorris :) )
    """
    def __init__(self, playerID=0):
        # Player has an ID that is either 0 or 1
        if playerID in ["a", "b"]:
            playerID = {"a": 0, "b": 1}[playerID]
        assert playerID in [0, 1]
        self.playerID = playerID

    # This is the method to implement in 'real' player-classes.
    # It takes a board and the NineMenMorris-phase and should in the implementation return a legal move.
    def get_move(self, phase, board: Board):
        assert phase in ["set", "move", "jump", "take"]
        raise NotImplementedError()

    # What the player does when it wins. Can change that in players.
    def win(self):
        print("Player {}: I won :)".format(self.playerID))


# Just parsing a string with some (>= 3) numbers in it to the numbers (only the first 3)
def parse_point(string):
    #print("point stuff:")
    #print([int(s) for s in re.findall(r'\d+', string)])
    r, x, y = [int(s) for s in re.findall(r'\d+', string)][:3]
    return r, x, y


class CmdPlayer(PlayerI):
    """
        The first and most simple implementation of a player. Just asks the user what to do on the command line.
    """
    def __init__(self, playerID=0):
        # Call this first in your implementations of the player also.
        super().__init__(playerID)

    # The function we have this class for
    def get_move(self, phase, board: Board):
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
        self.net = net

    def get_move(self, phase, board: Board, eps=0):
        return super(NetPlayerI, self).get_move(phase, board)

    def win(self):
        super(NetPlayerI, self).win()


class QNetPlayer(NetPlayerI):
    def __init__(self, net, playerID=0):
        """
        :param net: R^103 -> R
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
        super(QNetPlayer, self).__init__(net, playerID)

    def get_move(self, phase, board: Board, eps=0.0):
        legal_moves = board.legal_moves(phase, self.playerID)
        if len(legal_moves) == 0:
            return None
        if random() < eps:
            return sample(legal_moves, 1)[0]
        max_q = -2 ** 62
        max_action = None
        for move in legal_moves:
            encoded = QEncoding()(move, board, phase, self.playerID)
            q_val = self.net(encoded)[0][0]
            if q_val > max_q:
                max_q = q_val
                max_action = move
        return max_action

    def win(self):
        pass


class VNetPlayer(NetPlayerI):
    def __init__(self, net, playerID=0):
        """
        We want to estimate V(s) = max_a Q(s,a)

        input vector: phases in R^4 x board in R^24*2 (my positions x enemy positions)

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

        :param net: input vector in R^52 to R
        :param playerID: in [0,1]
        """
        super(VNetPlayer, self).__init__(net, playerID)

    def get_move(self, phase, board: Board, eps=0):
        legal_moves = board.legal_moves(phase, self.playerID)
        if len(legal_moves) == 0:
            return None
        if random() < eps:
            return sample(legal_moves, 1)[0]
        max_v = -2 ** 62
        max_action = None
        for move in legal_moves:
            # simulate move:
            sim_board = deepcopy(board)
            sim_board.do(move, self.playerID)
            encoded = VEncoding()(None, sim_board, phase, self.playerID)
            q_val = self.net(encoded)[0][0]
            if q_val > max_v:
                max_v = q_val
                max_action = move
        return max_action

    def win(self):
        pass


class RandomPlayer(PlayerI):
    """
    This player just does random moves only. Used for evaluation.
    """
    def __init__(self, playerID=0):
        super(RandomPlayer, self).__init__(playerID)

    def get_move(self, phase, board: Board):
        legal = board.legal_moves(phase, self.playerID)
        if len(legal) == 0:
            return None
        return sample(legal, 1)[0]

    def win(self):
        pass
