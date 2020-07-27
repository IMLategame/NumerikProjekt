from copy import deepcopy

from TicTacToe.board import Board
from TicTacToe.players import PlayerI, NetPlayerI
from network_backend.reinforcement_learning.rewardFunctions import RewardI
from network_backend.reinforcement_learning.utils import ReplayMem


class Game:
    """
        TicTacToe NineMenMorris logic.
    """
    def __init__(self, p0: PlayerI, p1: PlayerI, run=True, mem: ReplayMem=None, reward: RewardI=None):
        self.p0 = p0
        self.p1 = p1
        self.mem = mem
        self.reward = reward
        self.board = Board()
        self.winner = None
        if mem is not None:
            assert reward is not None
        if run:
            self.play()

    def getMove(self, player: PlayerI, eps=0.0):
        while True:
            if isinstance(player, NetPlayerI):
                move = player.get_move(board=self.board, eps=eps)
            else:
                move = player.get_move(board=self.board)
            if self.board.is_legal(move, player.playerID):
                return move
            print(player.playerID, self.board)

    def get_and_do_move(self, player: PlayerI, eps=0.0):
        move = self.getMove(player, eps)
        assert move in range(9)
        self.board.do(move, player.playerID)
        return move

    def learn(self, eps=0.1):
        assert self.mem is not None
        p0_is_net = isinstance(self.p0, NetPlayerI)
        p1_is_net = isinstance(self.p1, NetPlayerI)
        last_state_p0 = None
        last_state_p1 = None
        self.winner = None
        self.board.clear()

        moves = 0

        while not self.board.is_terminal(self.p0.playerID):
            move = self.get_and_do_move(self.p0, eps)
            if last_state_p0 is not None and p0_is_net:
                board_prev = last_state_p0
                board_post = deepcopy(self.board)
                r = self.reward(board_prev, board_post, None, self.p0.playerID)
                self.mem.add(board_prev, None, move, r, board_post, None, self.p0.playerID)
            last_state_p0 = deepcopy(self.board)
            moves += 1
            if self.board.is_terminal(self.p1.playerID):
                break
            move = self.get_and_do_move(self.p1, eps)
            if last_state_p1 is not None and p1_is_net:
                board_prev = last_state_p1
                board_post = deepcopy(self.board)
                r = self.reward(board_prev, board_post, None, self.p1.playerID)
                self.mem.add(board_prev, None, move, r, board_post, None, self.p1.playerID)
            last_state_p1 = deepcopy(self.board)
            moves += 1
        if len(self.board.get_rows(self.p0.playerID)) > 0:
            self.p0.win()
            self.winner = self.p0
        elif len(self.board.get_rows(self.p1.playerID)) > 0:
            self.p1.win()
            self.winner = self.p1
        return moves

    def play(self, wait_and_show=False, max_moves=None):
        self.board.clear()
        self.winner = None

        while not self.board.is_terminal(self.p0.playerID):
            move = self.get_and_do_move(self.p0)
            if wait_and_show:
                print(self.board)
                input("continue?")
            if self.board.is_terminal(self.p1.playerID):
                break
            if wait_and_show:
                print(self.board)
                input("continue?")
            move = self.get_and_do_move(self.p1)
        if len(self.board.get_rows(self.p0.playerID)) > 0:
            self.p0.win()
            self.winner = self.p0
            return
        if len(self.board.get_rows(self.p1.playerID)) > 0:
            self.p1.win()
            self.winner = self.p1
