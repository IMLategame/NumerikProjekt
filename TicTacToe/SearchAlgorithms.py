from copy import deepcopy
from math import sqrt, log, pow
from random import sample, random
from numpy.random import dirichlet


def miniMax(board, playerId):
    """
        Basic MiniMax
        :param board: current Board
        :param playerId: player whos turn it is
        :return: best move, value of best move
    """
    if board.is_terminal(playerId):
        if board.winner is None:
            return -1, 0.5
        if board.winner == board.player_map[playerId]:
            return -1, 1.0
        return -1, 0.0
    max_val = -2 ** 62
    max_action = None
    for a in board.legal_moves():
        simulated = deepcopy(board)
        simulated.do(a, playerId)
        mv, val = miniMax(simulated, 1 - playerId)
        if 1.0 - val > max_val:
            max_val = 1.0 - val
            max_action = a
    return max_action, max_val


class MCTSActionMemory:
    def __init__(self):
        self.mem_state = {}

    def __setitem__(self, key, value):
        state, player, action = key
        self.mem_state[state][player][action] = value

    def __getitem__(self, item):
        state, player, action = item
        return self.mem_state[state][player][action]

    def has(self, state, player):
        if state not in self.mem_state:
            return False
        if player not in self.mem_state[state]:
            return False
        return True

    def expand_into(self, state, player):
        assert not self.has(state, player)
        cpy = deepcopy(state)
        self.mem_state[cpy] = {}
        self.mem_state[cpy][player] = [0.0 for _ in range(9)]


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


class MCTSGuideI:
    def distr(self, state, player, action):
        raise NotImplementedError()

    def val(self, state, player):
        raise NotImplementedError()

    def possible_moves(self):
        raise NotImplementedError()


class GuidedMCTS:
    """
        MCTS algorithm with guiding mechanism.
    """
    def __init__(self, guide: MCTSGuideI, inv_temp=1/50, c=5, simulations=100, alpha=1.8, eps=0.25):
        self.guide = guide
        self.sum_qs = MCTSActionMemory()
        self.ns = MCTSActionMemory()
        self.inv_temp = inv_temp
        self.c = c
        self.simulations = simulations
        self.alpha = alpha
        self.eps = eps
        self.noise = [len(guide.possible_moves()) for _ in guide.possible_moves()]

    def utility(self, state, player, action, is_root=False):
        sum_q = self.sum_qs[(state, player, action)]
        p = self.guide.distr(state, player, action)
        if is_root:
            p = (1.0 - self.eps) * p + self.eps * self.noise[action]
        assert 0 <= p <= 1
        n = self.ns[(state, player, action)]
        N = 0
        for a in state.legal_moves(player):
            N += self.ns[(state, player, a)]
        if n == 0:
            return p * sqrt(N)
        return sum_q / n + self.c * p * sqrt(N) / (1 + n)

    def selection(self, root, player_turn, visited=set(), is_root=False):
        """
            Selection part
            :param root: node to start selection
            :param player_turn: current player
            :return: list of selected nodes and players
        """
        if not self.ns.has(root, player_turn):
            return [[root, player_turn, None]]
        max_util = -2 ** 62
        max_action = None
        max_state = None
        # select legal move with the maximum utility
        for a in root.legal_moves(player_turn):
            u = self.utility(root, player_turn, a, is_root=is_root)
            if u > max_util:
                simulated = deepcopy(root)
                simulated.do(a, player_turn)
                # do not visit the same state twice in order to not start an endless loop
                if (simulated, player_turn) in visited:
                    continue
                max_util = u
                max_action = a
                max_state = simulated
        #visited.add((deepcopy(root), player_turn))
        if max_state is None:
            return [[root, player_turn, None]]
        return [[root, player_turn, max_action]] + self.selection(max_state, 1 - player_turn, visited)

    def expansion(self, leaf, player_turn):
        """
            Expands a leaf node and adds it to the transition memory
            :param leaf: unexpanded node
            :param player_turn: current player
            :return: a child of the node or none
        """
        if leaf.is_terminal(player_turn):
            return
        self.ns.expand_into(leaf, player_turn)
        self.sum_qs.expand_into(leaf, player_turn)

    def simulation(self, start_node, player_turn):
        """
            Simulation step.
            :param start_node: start point of simulation
            :param player_turn: player whos turn it is at the start
            :return: estimated value of the game
        """
        return self.guide.val(start_node, player_turn)

    def backpropagation(self, nodes, value, player_turn):
        for state, player, move in nodes[:-1]:
            self.ns[(state, player, move)] += 1
            if player == player_turn:
                self.sum_qs[(state, player, move)] += value
            else:
                self.sum_qs[(state, player, move)] -= value

    def get_distr(self, board, player):
        N = 0
        move_map = {}
        for move in self.guide.possible_moves():
            n = pow(self.ns[(board, player, move)], 1.0/self.inv_temp)
            move_map[move] = n
            N += n
            if move not in board.legal_moves(player):
                assert n == 0
        assert N != 0
        for move in self.guide.possible_moves():
            move_map[move] *= 1.0/N
        return move_map

    def sample(self, d):
        u = random()
        for a in d:
            if d[a] <= 0.0:
                continue
            if d[a] > u:
                return a
            u -= d[a]

    def __call__(self, board, player):
        # sample noise for the root node
        self.noise = dirichlet([self.alpha for _ in self.guide.possible_moves()])
        for _ in range(self.simulations+1):
            sel_list = self.selection(board, player, is_root=True)
            self.expansion(sel_list[-1][0], sel_list[-1][1])
            val = self.simulation(sel_list[-1][0], sel_list[-1][1])
            if sel_list[-1][1] != player:
                val = -val
            self.backpropagation(sel_list, val, player)
        # sample move from the new distribution
        dist = self.get_distr(board, player)
        move = self.sample(dist)
        return move, dist

    def reset(self):
        self.ns.reset()
        self.sum_qs.reset()


class MCTSGuide(MCTSGuideI):
    def distr(self, state, player, action):
        return 1

    def val(self, state, player):
        # simulate:
        if state.is_terminal(player):
            if state.winner is None:
                return 0.0
            if state.winner == state.player_map[player]:
                return 1.0
            return -1.0
        action = sample(state.legal_moves(player), 1)[0]
        simulated = deepcopy(state)
        simulated.do(action, player)
        return self.val(simulated, 1-player)

    def possible_moves(self):
        return range(9)


class MCTS(GuidedMCTS):
    def __init__(self):
        super(MCTS, self).__init__(MCTSGuide(), simulations=100, inv_temp=1/70, eps=0.0, c=1.5)
