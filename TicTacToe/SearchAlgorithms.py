from copy import deepcopy


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


def MCTS(board, player):
    """
        Monte-Carlo-Tree-Search
        :param board: current board state
        :param player: player whos turn it is
        :return:
    """
