from copy import deepcopy
from math import sqrt, log
from random import sample

from NineMenMorris.board import Board


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


class MCTS:
    """
        Monte-Carlo-Tree-Search
    """
    def __init__(self, c=sqrt(2), simulations=100):
        self.visit_memory = {}
        self.c = c
        self.simulations = simulations

    def children(self, node, phase, player_turn):
        children = []
        for a in node.legal_moves(phase, player_turn):
            simulated = deepcopy(node)
            simulated.do(a, player_turn)
            next_phase, next_player = phase_and_player_after_sim(phase, a, simulated, player_turn)
            if simulated not in self.visit_memory:
                self.visit_memory[simulated] = {}
            if next_phase not in self.visit_memory[simulated]:
                self.visit_memory[simulated][next_phase] = {}
            if next_player not in self.visit_memory[simulated][next_phase]:
                self.visit_memory[simulated][next_phase][next_player] = [0, 0]
            children.append((simulated, next_phase, next_player))
        return children

    def utility(self, w, n, N):
        if n <= 0:
            return 2**62
        return w/n + self.c * sqrt(log(N, 10)/n)

    def selection(self, root, phase, player_turn):
        """
            Selection part
            :param root: node to start selection
            :param player_turn: current player
            :return: list of selected nodes and players
        """
        selection_list = [(root, phase, player_turn)]
        if root.is_terminal(phase, player_turn):
            return selection_list
        if root not in self.visit_memory:
            return selection_list
        if phase not in self.visit_memory[root]:
            return selection_list
        if player_turn not in self.visit_memory[root][phase]:
            return selection_list
        _, N = self.visit_memory[root][phase][player_turn]
        if N == 0:
            return selection_list
        max_util = -2**62
        max_state = None
        max_player = None
        for child, next_phase, next_player in self.children(root, phase, player_turn):
            w, n = self.visit_memory[child][next_phase][next_player]
            u = self.utility(w, n, N)
            if u > max_util:
                max_util = u
                max_state = child
                max_player = next_player
        return selection_list + self.selection(max_state, next_phase, max_player)

    def expansion(self, leaf, phase, player_turn):
        """
            Expands a leaf node and adds it to the transition memory
            :param leaf: unexpanded node
            :param player_turn: current player
            :return: a child of the node or none
        """
        if leaf.is_terminal(phase, player_turn):
            return None, None, None
        if leaf not in self.visit_memory:
            self.visit_memory[deepcopy(leaf)] = {}
        if phase not in self.visit_memory[leaf]:
            self.visit_memory[leaf][phase] = {}
        if player_turn not in self.visit_memory[leaf][phase]:
            self.visit_memory[leaf][phase][player_turn] = [0, 0]
        for move in leaf.legal_moves(phase, player_turn):
            simulated = deepcopy(leaf)
            simulated.do(move, player_turn)
            next_phase, next_player = phase_and_player_after_sim(phase, move, simulated, player_turn)
            if simulated not in self.visit_memory:
                self.visit_memory[simulated] = {}
            if next_phase not in self.visit_memory[simulated]:
                self.visit_memory[simulated][next_phase] = {}
            if next_player not in self.visit_memory[simulated][next_phase]:
                self.visit_memory[simulated][next_phase][next_player] = [0, 0]
        move = sample(leaf.legal_moves(phase, player_turn), 1)[0]
        simulated = deepcopy(leaf)
        simulated.do(move, player_turn)
        next_phase, next_player = phase_and_player_after_sim(phase, move, simulated, player_turn)
        if simulated not in self.visit_memory:
            self.visit_memory[simulated] = {}
        if next_phase not in self.visit_memory[simulated]:
            self.visit_memory[simulated][next_phase] = {}
        if next_player not in self.visit_memory[simulated][next_phase]:
            self.visit_memory[simulated][next_phase][next_player] = [0, 0]
        return simulated, next_phase, next_player

    def simulation(self, start_node, phase, player_turn):
        """
            Simulation step.
            :param start_node: start point of simulation
            :param player_turn: player whos turn it is at the start
            :return: winner
        """
        player = player_turn
        phase = phase
        node = deepcopy(start_node)
        depth = 0
        while not node.is_terminal(phase, player):
            move = sample(node.legal_moves(phase, player), 1)[0]
            node.do(move, player)
            phase, player = phase_and_player_after_sim(phase, move, node, player)
            depth += 1
            if depth > 1000:
                return None
        return node.winner

    def backpropagation(self, nodes, winner):
        if winner is None:
            val = 0.5
        elif winner == nodes[-1][0].player_map[nodes[-1][2]]:
            val = 1
        else:
            val = 0
        for node, phase, player in nodes:
            self.visit_memory[node][phase][player][0] += val
            self.visit_memory[node][phase][player][1] += 1

    def __call__(self, board, phase, player):
        if board not in self.visit_memory:
            self.visit_memory[board] = {}
        if phase not in self.visit_memory[board]:
            self.visit_memory[board][phase] = {}
        if player not in self.visit_memory[board][phase]:
            self.visit_memory[board][phase][player] = [0, 0]
        for _ in range(self.simulations):
            print("GO SIMULATION {}".format(_))
            sel_list = self.selection(board, phase, player)
            next_node, next_phase, next_player = self.expansion(sel_list[-1][0], sel_list[-1][1], sel_list[-1][2])
            if next_node is None:
                next_node = sel_list[-1][0]
                next_phase = sel_list[-1][1]
                next_player = sel_list[-1][2]
            else:
                sel_list.append((next_node, next_phase, next_player))
            winner = self.simulation(next_node, next_phase, next_player)
            self.backpropagation(sel_list, winner)
        max_util = -2**62
        max_a = None
        _, N = self.visit_memory[board][phase][player]
        for a in board.legal_moves(phase, player):
            simulated = deepcopy(board)
            simulated.do(a, player)
            next_phase, next_player = phase_and_player_after_sim(phase, a, simulated, player)
            w, n = self.visit_memory[simulated][next_phase][next_player]
            u = n
            if u > max_util:
                max_util = u
                max_a = a
        return max_a
