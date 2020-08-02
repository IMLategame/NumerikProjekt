from NineMenMorris.board import Board
from NineMenMorris.players import PlayerI, CmdPlayer, NetPlayerI
from copy import deepcopy
from network_backend.reinforcement_learning.utils import ReplayMem
from network_backend.reinforcement_learning.rewardFunctions import RewardI


class Game:
    """
        Class for the NineMenMorris logic.
        Specify the players and if you want to start the game immediately.
    """
    def __init__(self, run=True, p0=CmdPlayer(0), p1=CmdPlayer(1), mem: ReplayMem = None, reward: RewardI = None):
        self.p0 = p0
        self.p1 = p1
        self.board = Board()
        self.no_pieces_set = 9
        self.mem = mem
        self.reward = reward
        self.winner = None
        if mem is not None:
            assert reward is not None
        if run:
            self.play()

    # Get a move from the player in the phase and check if its legal.
    def getMove(self, player: PlayerI, phase, eps=0.0):
        while True:
            if isinstance(player, NetPlayerI):
                move = player.get_move(phase, self.board, eps)
            else:
                move = player.get_move(phase, self.board)
            if self.board.is_legal(move, phase, player.playerID):
                return move
            print(player.playerID, phase, self.board)

    def get_and_do_move(self, player: PlayerI, phase, eps=0.0):
        move = self.getMove(player, phase, eps)
        self.board.do(move, player.playerID)
        return move

    def learn(self, eps=0.1, max_steps=300):
        assert self.mem is not None
        assert isinstance(self.p0, NetPlayerI) or isinstance(self.p1, NetPlayerI)
        self.board.clear()
        p0_is_net = isinstance(self.p0, NetPlayerI)
        p1_is_net = isinstance(self.p1, NetPlayerI)
        last_state_p0 = None
        last_state_p1 = None
        moves = 0
        # set phase
        for _ in range(self.no_pieces_set):
            # p0
            move = self.get_and_do_move(self.p0, "set", eps)
            if last_state_p0 is not None and p0_is_net:
                board_prev = last_state_p0[0]
                phase_prev = last_state_p0[1]
                board_post = deepcopy(self.board)
                r = self.reward(board_prev, board_post, phase_prev, self.p0.playerID)
                self.mem.add(board_prev, phase_prev, move, r, board_post, "set", self.p0.playerID)
            last_state_p0 = (deepcopy(self.board), "set")

            # p1
            move = self.get_and_do_move(self.p1, "set", eps)
            self.board.do(move, self.p1.playerID)
            if last_state_p1 is not None and p1_is_net:
                board_prev = last_state_p1[0]
                phase_prev = last_state_p1[1]
                board_post = deepcopy(self.board)
                r = self.reward(board_prev, board_post, phase_prev, self.p1.playerID)
                self.mem.add(board_prev, phase_prev, move, r, board_post, "set", self.p1.playerID)
            last_state_p1 = (deepcopy(self.board), "set")
            moves += 2
        # move phase
        while len(self.board.get_player_pos(self.p0.playerID)) > 2 and len(
                self.board.get_player_pos(self.p1.playerID)) > 2 \
                and len(self.board.legal_moves("move", self.p0.playerID)) > 0\
                and moves < max_steps:
            # p0
            if len(self.board.get_player_pos(self.p0.playerID)) == 3:
                move = self.get_and_do_move(self.p0, "jump", eps)
            else:
                move = self.get_and_do_move(self.p0, "move", eps)
            if last_state_p0 is not None and p0_is_net:
                board_prev = last_state_p0[0]
                phase_prev = last_state_p0[1]
                board_post = deepcopy(self.board)
                r = self.reward(board_prev, board_post, phase_prev, self.p0.playerID)
                self.mem.add(board_prev, phase_prev, move, r, board_post, "move", self.p0.playerID)
            last_state_p0 = (deepcopy(self.board), "move")
            moves += 1
            if self.board.in_mull(self.p0.playerID, move.end):
                self.get_and_do_move(self.p0, "take", eps)
                if last_state_p0 is not None and p0_is_net:
                    board_prev = last_state_p0[0]
                    phase_prev = last_state_p0[1]
                    board_post = deepcopy(self.board)
                    r = self.reward(board_prev, board_post, phase_prev, self.p0.playerID)
                    self.mem.add(board_prev, phase_prev, move, r, board_post, "take", self.p0.playerID)
                last_state_p0 = (deepcopy(self.board), "take")
                moves += 1
                if len(self.board.get_player_pos(self.p1.playerID)) <= 2:
                    continue

            if len(self.board.legal_moves("move", self.p1.playerID)) == 0:
                break
            # p1
            if len(self.board.get_player_pos(self.p1.playerID)) == 3:
                move = self.get_and_do_move(self.p1, "jump", eps)
            else:
                move = self.get_and_do_move(self.p1, "move", eps)
            if last_state_p1 is not None and p1_is_net:
                board_prev = last_state_p1[0]
                phase_prev = last_state_p1[1]
                board_post = deepcopy(self.board)
                r = self.reward(board_prev, board_post, phase_prev, self.p1.playerID)
                self.mem.add(board_prev, phase_prev, move, r, board_post, "move", self.p1.playerID)
            last_state_p1 = (deepcopy(self.board), "move")
            moves += 1
            if self.board.in_mull(self.p1.playerID, move.end):
                self.get_and_do_move(self.p1, "take", eps)
                if last_state_p1 is not None and p1_is_net:
                    board_prev = last_state_p1[0]
                    phase_prev = last_state_p1[1]
                    board_post = deepcopy(self.board)
                    r = self.reward(board_prev, board_post, phase_prev, self.p1.playerID)
                    self.mem.add(board_prev, phase_prev, move, r, board_post, "take", self.p1.playerID)
                last_state_p1 = (deepcopy(self.board), "take")
                moves += 1
        # check winner
        if len(self.board.get_player_pos(self.p0.playerID)) > 2 or len(self.board.legal_moves("move", self.p1.playerID)) == 0:
            self.p0.win()
            self.winner = self.p0
        else:
            self.p1.win()
            self.winner = self.p1
        return moves

    # Its playtime!
    # Just get moves from players and perform them on the board.
    def play(self, wait_and_show=False, max_moves=2**62):
        self.board.clear()
        # set phase
        for _ in range(self.no_pieces_set):
            move = self.get_and_do_move(self.p0, "set")
            self.board.do(move, self.p0.playerID)
            if wait_and_show:
                print(self.board)
                input("continue?")
            move = self.get_and_do_move(self.p1, "set")
            self.board.do(move, self.p1.playerID)
            if wait_and_show:
                print(self.board)
                input("continue?")
        moves = 0
        # move phase
        while len(self.board.get_player_pos(self.p0.playerID)) > 2 and len(
                self.board.get_player_pos(self.p1.playerID)) > 2 \
                and len(self.board.legal_moves("move", self.p0.playerID)) > 0\
                and moves<max_moves:
            if len(self.board.get_player_pos(self.p0.playerID)) == 3:
                move = self.get_and_do_move(self.p0, "jump")
            else:
                move = self.get_and_do_move(self.p0, "move")
            if wait_and_show:
                print(self.board)
                input("continue?")
            if self.board.in_mull(self.p0.playerID, move.end):
                self.get_and_do_move(self.p0, "take")
                if wait_and_show:
                    print(self.board)
                    input("continue?")
                if len(self.board.get_player_pos(self.p1.playerID)) <= 2:
                    continue
            if len(self.board.legal_moves("move", self.p1.playerID)) == 0:
                break
            if len(self.board.get_player_pos(self.p1.playerID)) == 3:
                move = self.get_and_do_move(self.p1, "jump")
            else:
                move = self.get_and_do_move(self.p1, "move")
            if wait_and_show:
                print(self.board)
                input("continue?")
            if self.board.in_mull(self.p1.playerID, move.end):
                self.get_and_do_move(self.p1, "take")
                if wait_and_show:
                    print(self.board)
                    input("continue?")
            moves += 1
        # check winner
        if len(self.board.get_player_pos(self.p1.playerID)) <= 2 or len(
                self.board.legal_moves("move", self.p1.playerID)) == 0:
            self.p0.win()
            self.winner = self.p0
        else:
            self.p1.win()
            self.winner = self.p1
        self.p0.end(self.board)
        self.p1.end(self.board)
