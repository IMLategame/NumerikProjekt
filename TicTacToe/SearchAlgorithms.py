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


class MCTS:
    """
        Monte-Carlo-Tree-Search
    """

    def __init__(self, c=sqrt(2), simulations=100):
        self.visit_memory = {}
        self.ws = MCTSMemory()
        self.ns = MCTSMemory()
        self.c = c
        self.simulations = simulations

    def children(self, node, player_turn):
        children = []
        for a in node.legal_moves(player_turn):
            simulated = deepcopy(node)
            simulated.do(a, player_turn)
            children.append(simulated)
        return children

    def utility(self, w, n, N):
        if n <= 0:
            return 2 ** 62
        return w / n + self.c * sqrt(log(N, 10) / n)

    def selection(self, root, player_turn):
        """
            Selection part
            :param root: node to start selection
            :param player_turn: current player
            :return: list of selected nodes and players
        """
        selection_list = [(root, player_turn)]
        if root.is_terminal(player_turn):
            return selection_list
        N = self.ns[(root, player_turn)]
        if N == 0:
            return selection_list
        max_util = -2 ** 62
        max_state = None
        for child in self.children(root, player_turn):
            w = self.ws[(child, 1 - player_turn)]
            n = self.ns[(child, 1 - player_turn)]
            u = self.utility(w, n, N)
            if u > max_util:
                max_util = u
                max_state = child
        return selection_list + self.selection(max_state, 1 - player_turn)

    def expansion(self, leaf, player_turn):
        """
            Expands a leaf node and adds it to the transition memory
            :param leaf: unexpanded node
            :param player_turn: current player
            :return: a child of the node or none
        """
        if leaf.is_terminal(player_turn):
            return None
        for move in leaf.legal_moves(player_turn):
            simulated = deepcopy(leaf)
            simulated.do(move, player_turn)
        move = sample(leaf.legal_moves(player_turn), 1)[0]
        simulated = deepcopy(leaf)
        simulated.do(move, player_turn)
        return simulated

    def simulation(self, start_node, player_turn):
        """
            Simulation step.
            :param start_node: start point of simulation
            :param player_turn: player whos turn it is at the start
            :return: winner
        """
        player = player_turn
        node = deepcopy(start_node)
        while not node.is_terminal(player):
            move = sample(node.legal_moves(player), 1)[0]
            node.do(move, player)
            player = 1 - player
        return node.winner

    def backpropagation(self, nodes, winner, playerID):
        if winner is None:
            val = 0.5
        elif winner == nodes[-1][0].player_map[playerID]:
            val = 1
        else:
            val = 0
        for node, player in nodes:
            self.ns[node, player] += 1
            self.ws[node, player] += val

    def __call__(self, board, player):
        for _ in range(self.simulations):
            sel_list = self.selection(board, player)
            next_node = self.expansion(sel_list[-1][0], sel_list[-1][1])
            if next_node is None:
                next_node = sel_list[-1][0]
            else:
                sel_list.append((next_node, 1 - sel_list[-1][1]))
            winner = self.simulation(next_node, sel_list[-1][1])
            self.backpropagation(sel_list, winner, player)
        max_util = -2 ** 62
        max_a = None
        for a in board.legal_moves(player):
            simulated = deepcopy(board)
            simulated.do(a, player)
            n = self.ns[(simulated, 1 - player)]
            u = n
            if u > max_util:
                max_util = u
                max_a = a
        return max_a


class MCTSMemory:
    """
        To manage holding the data more easily.
    """

    def __init__(self):
        self.mem_state = {}

    def __setitem__(self, key, value):
        state, player = key
        if state not in self.mem_state:
            self.mem_state[state] = {}
        if player not in self.mem_state[state]:
            self.mem_state[state][player] = 0
        self.mem_state[state][player] = value

    def __getitem__(self, item):
        state, player = item
        if state not in self.mem_state:
            self.mem_state[state] = {}
        if player not in self.mem_state[state]:
            self.mem_state[state][player] = 0
        return self.mem_state[state][player]

    def reset(self):
        self.mem_state = {}

    def __len__(self):
        size = 0
        for key in self.mem_state:
            size += len(self.mem_state[key])
        return size


class MCTSActionMemory:
    def __init__(self):
        self.mem_state = {}

    def __setitem__(self, key, value):
        state, player, action = key
        """if state not in self.mem_state:
            self.mem_state[deepcopy(state)] = {}
        if player not in self.mem_state[state]:
            self.mem_state[state][player] = {}
        if action not in self.mem_state[state][player]:
            self.mem_state[state][player][action] = 0"""
        self.mem_state[state][player][action] = value

    def __getitem__(self, item):
        state, player, action = item
        """if state not in self.mem_state:
            self.mem_state[state] = {}
        if player not in self.mem_state[state]:
            self.mem_state[state][player] = {}
        if action not in self.mem_state[state][player]:
            self.mem_state[state][player][action] = 0"""
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
    def __init__(self, guide: MCTSGuideI, inv_temp=1/10, c=sqrt(2), simulations=50, alpha=2, eps=0.1):
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
            p = (1 - self.eps) * p + self.eps * self.noise[action]
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
                # do not visit the same state twice in order to not get into an endless loop
                if (simulated, player_turn) in visited:
                    continue
                max_util = u
                max_action = a
                max_state = simulated
        # visited.add((deepcopy(root), player_turn))
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
            if not self.ns.has(leaf, player_turn):
                self.ns.expand_into(leaf, player_turn)
                self.sum_qs.expand_into(leaf, player_turn)
            return None
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

    def backpropagation(self, nodes, value):
        for state, player, move in nodes:
            if move is None:
                continue
            self.ns[(state, player, move)] += 1
            self.sum_qs[(state, player, move)] += value

    def get_distr(self, board, player):
        N = 0
        move_map = {}
        for move in self.guide.possible_moves():
            n = pow(self.ns[(board, player, move)], 1/self.inv_temp)
            move_map[move] = n
            N += n
            if move not in board.legal_moves(player):
                if n != 0:
                    print("DANGER: ")
                    print(board)
                    print(self.ns.mem_state[board][player])
                assert n == 0
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

    def __call__(self, board, player):
        # sample noise for the root node
        self.noise = dirichlet([self.alpha for _ in self.guide.possible_moves()])
        for _ in range(self.simulations):
            sel_list = self.selection(board, player, is_root=True)
            self.expansion(sel_list[-1][0], sel_list[-1][1])
            val = self.simulation(sel_list[-1][0], sel_list[-1][1])
            self.backpropagation(sel_list, val)
        # sample move from the new distribution
        dist = self.get_distr(board, player)
        move = self.sample(dist)
        return move, dist

    def reset(self):
        self.ns.reset()
        self.sum_qs.reset()
