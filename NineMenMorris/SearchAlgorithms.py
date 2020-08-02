from copy import deepcopy
from math import sqrt, log
from random import sample, random

from numpy.random.mtrand import dirichlet

from NineMenMorris.board import Board
from NineMenMorris.moves import Move


def miniMax(board: Board, phase, playerId):
    """
        Basic MiniMax
        :param board: current Board
        :param playerId: player whos turn it is
        :return: best move, value of best move
    """
    if board.is_terminal(phase, playerId):
        if board.winner is None:
            return -1, 0.5
        if board.winner == board.player_map[playerId]:
            return -1, 1.0
        return -1, 0.0
    max_val = -2**62
    max_action = None
    for a in board.legal_moves(phase=phase, player=playerId):
        simulated = deepcopy(board)
        simulated.do(a, playerId)
        next_phase, next_player = phase_and_player_after_sim(phase, a, simulated, playerId)
        mv, val = miniMax(simulated, next_phase, next_player)
        if 1.0-val > max_val:
            max_val = 1.0-val
            max_action = a
    return max_action, max_val


def phase_and_player_after_sim(phase, move, simulated, playerId):
    next_phase = None
    next_player = 1 - playerId
    if phase == "set":
        if len(simulated.get_player_pos(1-playerId)) >= 9:
            next_phase = "move"
        else:
            next_phase = "set"
    if phase == "move":
        if simulated.in_mull(playerId, move.end):
            next_phase = "take"
            next_player = playerId
        else:
            next_phase = "move"
    if phase == "take" or phase == "jump":
        if len(simulated.get_player_pos(1-playerId)) <= 3:
            next_phase = "jump"
        else:
            next_phase = "move"
    return next_phase, next_player


class MCTSGuideI:
    def distr(self, state, phase, player, action):
        raise NotImplementedError()

    def val(self, state, player):
        raise NotImplementedError()

    def possible_moves(self):
        raise NotImplementedError()


class MCTSActionMemory:
    def __init__(self, guide: MCTSGuideI):
        self.mem_state = {}
        self.guide = guide

    def __setitem__(self, key, value):
        state, phase, player, action = key
        self.mem_state[state][phase][player][action] = value

    def __getitem__(self, item):
        state, phase, player, action = item
        return self.mem_state[state][phase][player][action]

    def has(self, state, phase, player):
        if state not in self.mem_state:
            return False
        if phase not in self.mem_state[state]:
            return False
        if player not in self.mem_state[state][phase][player]:
            return False
        return True

    def expand_into(self, state, phase, player):
        assert not self.has(state, phase, player)
        cpy = deepcopy(state)
        self.mem_state[cpy] = {}
        self.mem_state[cpy][phase] = {}
        self.mem_state[cpy][phase][player] = {mv: 0.0 for mv in self.guide.possible_moves()}
        for state in self.mem_state:
            print(state)
            print(state not in self.mem_state)

    def reset(self):
        self.mem_state = {}

    def __str__(self):
        return str(self.mem_state)

    def sum(self):
        entries = 0
        for state in self.mem_state:
            for player in self.mem_state[state]:
                entries += sum(self.mem_state[state][player])
        return entries


class GuidedMCTS:
    """
        MCTS algorithm with guiding mechanism.
    """
    def __init__(self, guide: MCTSGuideI, inv_temp=1/50, c=5, simulations=50, alpha=1.8, eps=0.25):
        self.guide = guide
        self.sum_qs = MCTSActionMemory(guide)
        self.ns = MCTSActionMemory(guide)
        self.inv_temp = inv_temp
        self.c = c
        self.simulations = simulations
        self.alpha = alpha
        self.eps = eps
        self.noise = [1.0/len(guide.possible_moves()) for _ in guide.possible_moves()]
        self.move_to_idx = {mv: i for i, mv in enumerate(guide.possible_moves())}

    def utility(self, state, phase, player, action, is_root=False):
        sum_q = self.sum_qs[(state, phase, player, action)]
        p = self.guide.distr(state, phase, player, action)
        if is_root:
            noise = self.noise[self.move_to_idx[action]]
            p = (1 - self.eps) * p + self.eps * noise
        assert 0 <= p <= 1
        n = self.ns[(state, phase, player, action)]
        N = 0
        for a in state.legal_moves(phase, player):
            N += self.ns[(state, phase, player, a)]
        if n == 0:
            return p * sqrt(N)
        return sum_q / n + self.c * p * sqrt(N) / (1 + n)

    def selection(self, root, root_phase, player_turn, visited=set(), is_root=False):
        """
            Selection part
            :param root: node to start selection
            :param player_turn: current player
            :return: list of selected nodes and players
        """
        if not self.ns.has(root, root_phase, player_turn):
            return [[root, root_phase, player_turn, None]]
        max_util = -2 ** 62
        max_action = None
        max_state = None
        max_phase = None
        max_player = None
        # select legal move with the maximum utility
        for a in root.legal_moves(root_phase, player_turn):
            u = self.utility(root, root_phase, player_turn, a, is_root=is_root)
            if u > max_util:
                simulated = deepcopy(root)
                simulated.do(a, player_turn)
                # do not visit the same state twice in order to not get into an endless loop
                if (simulated, player_turn) in visited:
                    continue
                max_util = u
                max_action = a
                max_state = simulated
                max_phase, max_player = phase_and_player_after_sim(root_phase, a, simulated, player_turn)
        # visited.add((deepcopy(root), player_turn))
        assert max_state is not None
        return [[root, root_phase, player_turn, max_action]] + self.selection(max_state, max_phase, max_player, visited)

    def expansion(self, leaf, phase, player_turn):
        """
            Expands a leaf node and adds it to the transition memory
            :param leaf: unexpanded node
            :param player_turn: current player
            :return: a child of the node or none
        """
        if leaf.is_terminal(phase, player_turn):
            return
        self.ns.expand_into(leaf, phase, player_turn)
        self.sum_qs.expand_into(leaf, phase, player_turn)

    def simulation(self, start_node, start_phase, player_turn):
        """
            Simulation step.
            :param start_node: start point of simulation
            :param player_turn: player whos turn it is at the start
            :return: estimated value of the game
        """
        return self.guide.val(start_node, start_phase, player_turn)

    def backpropagation(self, nodes, value, player_turn):
        for state, phase, player, move in nodes[:-1]:
            self.ns[(state, phase, player, move)] += 1
            if player == player_turn:
                self.sum_qs[(state, phase, player, move)] += value
            else:
                self.sum_qs[(state, phase, player, move)] -= value

    def get_distr(self, board, phase, player):
        N = 0
        move_map = {}
        for move in self.guide.possible_moves():
            n = pow(self.ns[(board, phase, player, move)], 1/self.inv_temp)
            move_map[move] = n
            N += n
            if move not in board.legal_moves(phase, player):
                assert n == 0
        print(move_map)
        assert N != 0
        for move in self.guide.possible_moves():
            move_map[move] *= 1/N
        return move_map

    def sample(self, d):
        u = random()
        for a in d:
            if d[a] <= 0.0:
                continue
            if d[a] > u:
                return a
            u -= d[a]

    def __call__(self, board, phase, player):
        # sample noise for the root node
        self.noise = dirichlet([self.alpha for _ in self.guide.possible_moves()])
        for _ in range(self.simulations+1):
            print("Simulation {}".format(_), end="\r", flush=True)
            sel_list = self.selection(board, phase, player, is_root=True)
            self.expansion(sel_list[-1][0], sel_list[-1][1], sel_list[-1][2])
            try:
                val = self.simulation(sel_list[-1][0], sel_list[-1][1], sel_list[-1][2])
            except:
                continue
            if sel_list[-1][2] != player:
                val = -val
            print("BACKPROP", len(sel_list))
            self.backpropagation(sel_list, val, player)
        # sample move from the new distribution
        dist = self.get_distr(board, phase, player)
        move = self.sample(dist)
        return move, dist

    def reset(self):
        self.ns.reset()
        self.sum_qs.reset()


class MCTSGuide(MCTSGuideI):
    def distr(self, state, phase, player, action):
        return 1

    def val(self, state, phase, player, already_copied=False):
        # simulate:
        if state.is_terminal(phase, player):
            if state.winner is None:
                return 0.0
            if state.winner == state.player_map[player]:
                return 1.0
            return -1.0
        action = sample(state.legal_moves(phase, player), 1)[0]
        if not already_copied:
            simulated = deepcopy(state)
        else:
            simulated = state
        simulated.do(action, player)
        phase_post, player_post = phase_and_player_after_sim(phase, action, simulated, player)
        return self.val(simulated, phase_post, player_post, already_copied=True)

    def possible_moves(self):
        moves = []
        # phase = set:
        for x in range(3):
            for y in range(3):
                if x == 1 == y:
                    continue
                for r in range(3):
                    moves.append(Move("set", (r, x, y)))
        # phase = move / jump
        for x in range(3):
            for y in range(3):
                if x == 1 == y:
                    continue
                for r in range(3):
                    for x2 in range(3):
                        for y2 in range(3):
                            if x2 == 1 == y2:
                                continue
                            for r2 in range(3):
                                moves.append(Move("move", (r, x, y), (r2, x2, y2)))
        #phase = take
        for x in range(3):
            for y in range(3):
                if x == 1 == y:
                    continue
                for r in range(3):
                    moves.append(Move("take", (r, x, y)))
        return moves


class MCTS(GuidedMCTS):
    def __init__(self):
        super(MCTS, self).__init__(MCTSGuide(), simulations=50, inv_temp=1/100, eps=0.0)
