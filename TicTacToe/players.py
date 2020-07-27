from TicTacToe.board import Board
from random import random, sample

from network_backend.reinforcement_learning.encodings import TTTQEncoding


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
    def get_move(self, board: Board):
        raise NotImplementedError()

    # What the player does when it wins. Can change that in players.
    def win(self):
        print("Player {}: I won :)".format(self.playerID))


class NetPlayerI(PlayerI):
    def __init__(self, net, playerID=0):
        super(NetPlayerI, self).__init__(playerID)
        self.net = net

    def get_move(self, board: Board, eps=0):
        return super(NetPlayerI, self).get_move(board)

    def win(self):
        super(NetPlayerI, self).win()


class CmdPlayer(PlayerI):
    def get_move(self, board: Board):
        print("Its your turn. You are player {}".format(
            board.player_map[self.playerID]))
        print(board)
        while True:
            print("Whats your move? (move is an int in [0, 8]).")
            move_string = input("->")
            try:
                move = int(move_string)#
                if board.is_legal(move, self.playerID):
                    return move
            except:
                pass


class QNetPlayer(NetPlayerI):
    """
        This player gets a net for dimensions
        net: R^27 -> R
    """
    def __init__(self, net, playerID=0):
        super(QNetPlayer, self).__init__(net, playerID)
        self.moves = 0
        self.rand = 0

    def get_move(self, board: Board, eps=0.0):
        self.moves += 1
        legal_moves = board.legal_moves()
        if len(legal_moves) == 0:
            return None
        if random() < eps:
            self.rand += 1
            return sample(legal_moves, 1)[0]
        max_q = -2**62
        max_action = None
        for move in legal_moves:
            encoded = TTTQEncoding()(move, board, None, self.playerID)
            q_val = self.net(encoded)[0][0]
            if q_val > max_q:
                max_q = q_val
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

    def get_move(self, board: Board):
        legal = board.legal_moves(self.playerID)
        if len(legal) == 0:
            return None
        return sample(legal, 1)[0]

    def win(self):
        pass
