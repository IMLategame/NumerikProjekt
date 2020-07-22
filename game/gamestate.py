from game.board import Board
from game.players import PlayerI, CmdPlayer, NetPlayerI
from copy import deepcopy
from utils import ReplayMem
from network_backend.rewardFunctions import RewardI

"""
    Class for the game logic.
    Specify the players and if you want to start the game immediately.
"""
class Game:
    def __init__(self, run=True, p0 = CmdPlayer(0), p1 = CmdPlayer(1), mem: ReplayMem = None, reward: RewardI = None):
        self.p0 = p0
        self.p1 = p1
        self.board = Board()
        self.no_pieces_set = 9
        self.mem = mem
        self.reward = reward
        if mem is not None:
            assert reward is not None
        if run:
            self.play()

    # Get a move from the player in the phase and check if its legal.
    def getMove(self, player: PlayerI, phase):
        while True:
            move = player.getMove(phase, self.board)
            if self.board.is_legal(move, phase, player.playerID):
                return move

    def get_and_do_move(self, player: PlayerI, phase):
        #if self.mem is not None:
        #    state_prev = deepcopy(self.board)
        #    phase_prev = phase
        move = self.getMove(player, phase)
        self.board.do(move, player.playerID)
        #if self.mem is not None:
        #    state_post = deepcopy(self.board)
        #    # TODO: correct post phase if needed
        #    phase_post = phase
        #    r = self.reward(state_prev, state_post, phase_prev, player)
        #    self.mem.add(state_prev, phase_prev, move, r, state_post, phase_post, player.playerID)
        return move

    def learn(self):
        assert self.mem is not None
        assert isinstance(self.p0, NetPlayerI) or isinstance(self.p1, NetPlayerI)
        # set phase
        initial = True
        for _ in range(self.no_pieces_set):
            move = self.get_and_do_move(self.p0, "set")
            self.board.do(move, self.p0.playerID)
            move = self.get_and_do_move(self.p1, "set")
            self.board.do(move, self.p1.playerID)
        # move phase
        while len(self.board.get_player_pos(self.p0.playerID)) > 2 and len(
                self.board.get_player_pos(self.p1.playerID)) > 2:
            if len(self.board.get_player_pos(self.p0.playerID)) == 3:
                move = self.get_and_do_move(self.p0, "jump")
            else:
                move = self.get_and_do_move(self.p0, "move")
            if self.board.in_mull(self.p0.playerID, move.end):
                self.get_and_do_move(self.p0, "take")
            if len(self.board.get_player_pos(self.p1.playerID)) == 3:
                move = self.get_and_do_move(self.p1, "jump")
            else:
                move = self.get_and_do_move(self.p1, "move")
            if self.board.in_mull(self.p1.playerID, move.end):
                self.get_and_do_move(self.p1, "take")
        # check winner
        if len(self.board.get_player_pos(self.p0.playerID)) <= 2:
            self.p0.win()
        else:
            self.p1.win()

    # Its playtime!
    # Just get moves from players and perform them on the board.
    def play(self):
        # set phase
        for _ in range(self.no_pieces_set):
            move = self.get_and_do_move(self.p0, "set")
            self.board.do(move, self.p0.playerID)
            move = self.get_and_do_move(self.p1, "set")
            self.board.do(move, self.p1.playerID)
        # move phase
        while len(self.board.get_player_pos(self.p0.playerID)) > 2 and len(self.board.get_player_pos(self.p1.playerID)) > 2:
            if len(self.board.get_player_pos(self.p0.playerID)) == 3:
                move = self.get_and_do_move(self.p0, "jump")
            else:
                move = self.get_and_do_move(self.p0, "move")
            if self.board.in_mull(self.p0.playerID, move.end):
                self.get_and_do_move(self.p0, "take")
            if len(self.board.get_player_pos(self.p1.playerID)) == 3:
                move = self.get_and_do_move(self.p1, "jump")
            else:
                move = self.get_and_do_move(self.p1, "move")
            if self.board.in_mull(self.p1.playerID, move.end):
                self.get_and_do_move(self.p1, "take")
        # check winner
        if len(self.board.get_player_pos(self.p0.playerID)) <= 2:
            self.p0.win()
        else:
            self.p1.win()
