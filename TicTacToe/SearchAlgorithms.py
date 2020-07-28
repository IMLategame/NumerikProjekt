from copy import deepcopy
from math import sqrt, log
from random import sample


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


class MCTS:
    """
        Monte-Carlo-Tree-Search
    """
    def __init__(self, c=sqrt(2), simulations=500):
        self.visit_memory = {}
        self.c = c
        self.simulations = simulations

    def children(self, node, player_turn):
        children = []
        for a in node.legal_moves(player_turn):
            simulated = deepcopy(node)
            simulated.do(a, player_turn)
            if simulated not in self.visit_memory:
                self.visit_memory[simulated] = {}
            if 1-player_turn not in self.visit_memory[simulated]:
                self.visit_memory[simulated][1-player_turn] = [0, 0]
            children.append(simulated)
        return children

    def utility(self, w, n, N):
        if n <= 0:
            return 2**62
        return w/n + self.c * sqrt(log(N, 10)/n)

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
        if root not in self.visit_memory:
            return selection_list
        if player_turn not in self.visit_memory[root]:
            return selection_list
        _, N = self.visit_memory[root][player_turn]
        if N == 0:
            return selection_list
        max_util = -2**62
        max_state = None
        for child in self.children(root, player_turn):
            w, n = self.visit_memory[child][1-player_turn]
            u = self.utility(w, n, N)
            if u > max_util:
                max_util = u
                max_state = child
        return selection_list + self.selection(max_state, 1-player_turn)

    def expansion(self, leaf, player_turn):
        """
            Expands a leaf node and adds it to the transition memory
            :param leaf: unexpanded node
            :param player_turn: current player
            :return: a child of the node or none
        """
        if leaf.is_terminal(player_turn):
            return None
        if leaf not in self.visit_memory:
            self.visit_memory[deepcopy(leaf)] = {}
        if player_turn not in self.visit_memory[leaf]:
            self.visit_memory[leaf][player_turn] = [0, 0]
        for move in leaf.legal_moves(player_turn):
            simulated = deepcopy(leaf)
            simulated.do(move, player_turn)
            if simulated not in self.visit_memory:
                self.visit_memory[simulated] = {}
            if 1 - player_turn not in self.visit_memory[simulated]:
                self.visit_memory[simulated][1 - player_turn] = [0, 0]
        move = sample(leaf.legal_moves(player_turn), 1)[0]
        simulated = deepcopy(leaf)
        simulated.do(move, player_turn)
        if simulated not in self.visit_memory:
            self.visit_memory[simulated] = {}
        if 1-player_turn not in self.visit_memory[simulated]:
            self.visit_memory[simulated][1-player_turn] = [0, 0]
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
            player = 1-player
        return node.winner

    def backpropagation(self, nodes, winner, playerID):
        if winner is None:
            val = 0.5
        elif winner == nodes[-1][0].player_map[playerID]:
            val = 1
        else:
            val = 0
        for node, player in nodes:
            self.visit_memory[node][player][0] += val
            self.visit_memory[node][player][1] += 1

    def __call__(self, board, player):
        if board not in self.visit_memory:
            self.visit_memory[board] = {}
        if player not in self.visit_memory[board]:
            self.visit_memory[board][player] = [0, 0]
        for _ in range(self.simulations):
            sel_list = self.selection(board, player)
            next_node = self.expansion(sel_list[-1][0], sel_list[-1][1])
            if next_node is None:
                next_node = sel_list[-1][0]
            else:
                sel_list.append((next_node, 1-sel_list[-1][1]))
            winner = self.simulation(next_node, sel_list[-1][1])
            self.backpropagation(sel_list, winner, player)
        max_util = -2**62
        max_a = None
        _, N = self.visit_memory[board][player]
        for a in board.legal_moves(player):
            simulated = deepcopy(board)
            simulated.do(a, player)
            w, n = self.visit_memory[simulated][1-player]
            u = n
            if u > max_util:
                max_util = u
                max_a = a
        return max_a
