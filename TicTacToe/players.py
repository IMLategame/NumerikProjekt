from copy import deepcopy

from TicTacToe.board import Board
from random import random, sample
import pygame as pg

from network_backend.reinforcement_learning.encodings import TTTQEncoding, TTTVEncoding


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

    def end(self, board: Board):
        pass


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
                move = int(move_string)  #
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
        max_q = -2 ** 62
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


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class VisualPlayer(PlayerI):
    """
        This player draws the board in a separate window using pygame.
    """
    def __init__(self, playerID=0, size=300):
        super(VisualPlayer, self).__init__(playerID)
        pg.init()
        pg.font.init()
        self.font = pg.font.SysFont("Comic Sans MS", 90)
        self.size = size
        self.delta = 10
        self.screen = pg.display.set_mode((size, size))
        self.x_offset = 15
        self.y_offset = -18
        pg.display.set_caption("Tic Tac Toe")
        pg.mouse.set_visible(True)

    def draw_board(self, board: Board):
        self.screen.fill(WHITE)
        for i in range(2):
            i += 1
            pg.draw.line(self.screen, BLACK, (0, i * self.size / 3.0), (self.size, i * self.size / 3.0))
            pg.draw.line(self.screen, BLACK, (i * self.size / 3.0, 0), (i * self.size / 3.0, self.size))
        for x in range(3):
            for y in range(3):
                if board[x + 3 * y] != board.player_map[-1]:
                    text = self.font.render(board[x + 3 * y].capitalize(), False, BLACK)
                    self.screen.blit(text, (self.size / 3.0 * x + self.x_offset, self.size / 3.0 * y + self.y_offset))
        pg.display.flip()

    def get_move(self, board: Board):
        self.draw_board(board)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return None
        click_start_pos = None
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.MOUSEBUTTONDOWN:
                    click_start_pos = pg.mouse.get_pos()
                if event.type == pg.MOUSEBUTTONUP:
                    click_end_pos = pg.mouse.get_pos()
                    if click_start_pos is None:
                        continue
                    click_dist = abs(click_start_pos[0] - click_end_pos[0]) ** 2 \
                        + abs(click_start_pos[1] - click_end_pos[1]) ** 2
                    if click_dist < self.delta ** 2:
                        # get x val:
                        for x in range(3):
                            if click_start_pos[0] - self.delta >= self.size / 3.0 * x \
                                    and click_start_pos[0] + self.delta <= self.size / 3.0 * (x + 1):
                                for y in range(3):
                                    if click_start_pos[1] - self.delta >= self.size / 3.0 * y \
                                            and click_start_pos[1] + self.delta <= self.size / 3.0 * (y + 1):
                                        return 3 * y + x

    def end(self, board):
        self.draw_board(board)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT or event.type == pg.MOUSEBUTTONUP:
                    return


class VNetPlayer(NetPlayerI):
    """
        This player does deep value iteration.
        It takes a net: R^18 -> R
    """
    def get_move(self, board: Board, eps=0):
        legal_moves = board.legal_moves()
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
            encoded = TTTVEncoding()(None, sim_board, None, self.playerID)
            v_val = self.net(encoded)[0][0]
            if v_val > max_v:
                max_v = v_val
                max_action = move
        return max_action

    def win(self):
        pass


# Minimax algorithm:
def miniMax(board, playerId):
    if board.is_terminal(playerId):
        if board.winner is None:
            return -1, 0.5
        if board.winner == board.player_map[playerId]:
            return -1, 1.0
        return -1, 0.0
    max_val = -2**62
    max_action = None
    for a in board.legal_moves():
        simulated = deepcopy(board)
        simulated.do(a, playerId)
        mv, val = miniMax(simulated, 1-playerId)
        if 1.0-val > max_val:
            max_val = 1.0-val
            max_action = a
    return max_action, max_val


class MiniMaxPlayer(PlayerI):
    """
        This player runs a straight min max, expanding all possible future states.
    """
    def get_move(self, board: Board):
        mv, val = miniMax(board, self.playerID)
        return mv

    def win(self):
        pass
