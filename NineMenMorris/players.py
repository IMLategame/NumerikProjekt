from pygame.rect import Rect

from NineMenMorris.SearchAlgorithms import MCTS
from NineMenMorris.board import Board
from NineMenMorris.moves import Move
from network_backend.reinforcement_learning.encodings import QEncoding, VEncoding
import re
from random import random, sample
from copy import deepcopy
import pygame as pg


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

    # Called when the game has ended.
    def end(self, board: Board):
        pass


# Just parsing a string with some (>= 3) numbers in it to the numbers (only the first 3)
def parse_point(string):
    # print("point stuff:")
    # print([int(s) for s in re.findall(r'\d+', string)])
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


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GREY = (230, 230, 230)


class VisualPlayer(PlayerI):
    """
        This player draws the board for easier interaction.
    """

    def __init__(self, playerID=0):
        super(VisualPlayer, self).__init__(playerID)
        pg.init()
        pg.font.init()
        self.box_size = 50
        self.spacing = 25
        self.font = pg.font.SysFont("Comic Sans MS", 45)
        self.screen = pg.display.set_mode((7 * self.box_size + 6 * self.spacing, 7 * self.box_size + 6 * self.spacing))
        self.x_offset = 8
        self.y_offset = -7
        self.box_list = []
        pg.display.set_caption("Nine Men Morris")
        pg.mouse.set_visible(True)

    def draw_board(self, board: Board):
        self.screen.fill(WHITE)
        self.box_list = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if x == 0:
                        x_coord = (2 - r) * (self.box_size + self.spacing)
                    elif x == 1:
                        x_coord = 3 * (self.box_size + self.spacing)
                    elif x == 2:
                        x_coord = (4 + r) * (self.box_size + self.spacing)
                    if y == 0:
                        y_coord = (2 - r) * (self.box_size + self.spacing)
                    elif y == 1:
                        y_coord = 3 * (self.box_size + self.spacing)
                    elif y == 2:
                        y_coord = (4 + r) * (self.box_size + self.spacing)
                    # draw boxes
                    pg.draw.rect(self.screen, LIGHT_GREY, Rect(x_coord, y_coord, self.box_size, self.box_size))
                    # draw text in boxes
                    if board[r, x, y] != board.player_map[-1]:
                        text = self.font.render(board.string_rep[board[r, x, y]].capitalize(), False, BLACK)
                        self.screen.blit(text, (x_coord+self.x_offset, y_coord+self.y_offset))
                    self.box_list.append(((x_coord, y_coord), r, x, y))
                    # draw lines to the right
                    if (x < 2 or y == 1) and not (r == 0 == x and y == 1):
                        line_start_x_coord = x_coord + self.box_size
                        line_start_y_coord = y_coord + self.box_size / 2.0
                        line_end_x_coord = line_start_x_coord + r * (self.box_size + self.spacing) + self.spacing
                        if y == 1:
                            line_end_x_coord = line_start_x_coord + self.spacing
                        line_end_y_coord = line_start_y_coord
                        pg.draw.line(self.screen, BLACK, (line_start_x_coord, line_start_y_coord),
                                     (line_end_x_coord, line_end_y_coord))
                    # draw lines to the bottom
                    if (y < 2 or x == 1) and not (r == 0 == y and x == 1):
                        line_start_x_coord = x_coord + self.box_size / 2.0
                        line_start_y_coord = y_coord + self.box_size
                        line_end_y_coord = line_start_y_coord + r * (self.box_size + self.spacing) + self.spacing
                        if x == 1:
                            line_end_y_coord = line_start_y_coord + self.spacing
                        line_end_x_coord = line_start_x_coord
                        pg.draw.line(self.screen, BLACK, (line_start_x_coord, line_start_y_coord),
                                     (line_end_x_coord, line_end_y_coord))
        pg.display.flip()

    def get_move(self, phase, board: Board):
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
                    click_end_coord = None
                    for (box_x, box_y), r, x, y in self.box_list:
                        if box_x <= click_end_pos[0] <= box_x + self.box_size \
                                and box_y <= click_end_pos[1] <= box_y + self.box_size:
                            click_end_coord = (r, x, y)
                    if click_end_coord is None:
                        continue
                    if phase == "set":
                        return Move("set", click_end_coord)
                    if phase == "take":
                        return Move("take", click_end_coord)
                    click_start_coord = None
                    for (box_x, box_y), r, x, y in self.box_list:
                        if box_x <= click_start_pos[0] <= box_x + self.box_size \
                                and box_y <= click_start_pos[1] <= box_y + self.box_size:
                            click_start_coord = (r, x, y)
                    if click_start_coord is None:
                        continue
                    return Move("move", click_end_coord, click_start_coord)

    def end(self, board: Board):
        self.draw_board(board)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT or event.type == pg.MOUSEBUTTONUP:
                    return


class MCTSPlayer(PlayerI):
    def __init__(self, playerID):
        super(MCTSPlayer, self).__init__(playerID)
        self.mcts = MCTS()

    def get_move(self, phase, board: Board):
        return self.mcts(board, phase, self.playerID)

    def win(self):
        pass
