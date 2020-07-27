from copy import deepcopy


class Board:
    """
        Board encoding:

         0 | 1 | 2
        -----------
         3 | 4 | 5
        -----------
         6 | 7 | 8
    """
    def __init__(self, noP="_", p0="o", p1="x", board=None):
        # possible states:
        self.player_map = {-1: noP, 0: p0, 1: p1}
        self.board_state = [noP for _ in range(9)]
        if board is not None:
            self.board_state = deepcopy(board)

    def get_player_pos(self, player):
        assert player in [-1, 0, 1]
        player = self.player_map[player]
        pos = []
        for i in range(9):
            if self[i] == player:
                pos.append(i)
        return pos

    def get_empty_pos(self):
        return self.get_player_pos(-1)

    def __getitem__(self, item):
        assert item in range(9)
        return self.board_state[item]

    def __setitem__(self, key, value):
        assert key in range(9)
        self.board_state[key] = value

    def __str__(self):
        string = "\t {0[0]} | {0[1]} | {0[2]} \n" \
                 "\t-----------\n" \
                 "\t {0[3]} | {0[4]} | {0[5]} \n" \
                 "\t-----------\n" \
                 "\t {0[6]} | {0[7]} | {0[8]} \n".format(self)
        return string

    def legal_moves(self, dummy1=None, dummy2=None):
        return self.get_empty_pos()

    def is_legal(self, move, player=None):
        return move in self.get_empty_pos()

    def do(self, move, player):
        assert player in [-1, 0, 1]
        player = self.player_map[player]
        self[move] = player

    def get_rows(self, player):
        assert player in [-1, 0, 1]
        player = self.player_map[player]
        rows = []
        for i in range(3):
            if self[3*i] == self[3*i+1] and self[3*i+1] == self[3*i+2] and self[3*i+2] == player:
                rows.append((3*i, 3*i+1, 3*i+2))
            if self[i] == self[i+3] and self[i] == self[i+6] and self[i] == player:
                rows.append((i, i+3, i+6))
        if self[0] == self[4] and self[0] == self[8] and self[0] == player:
            rows.append((0, 4, 8))
        if self[2] == self[4] and self[4] == self[6] and self[2] == player:
            rows.append((2, 4, 6))
        return rows

    def in_row(self, player, point):
        rows = self.get_rows(player)
        for row in rows:
            if point in row:
                return True
        return False

    def is_terminal(self, player, phase=None):
        assert player in [-1, 0, 1]
        if len(self.get_rows(player)) > 0:
            self.winner = self.player_map[player]
            return True
        if len(self.get_rows(1-player)) > 0:
            self.winner = self.player_map[1-player]
            return True
        if len(self.get_empty_pos()) == 0:
            self.winner = None
            return True
        return False

    def clear(self):
        self.board_state = [self.player_map[-1] for _ in range(9)]

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        for key in self.player_map:
            if self.player_map[key] != other.player_map[key]:
                return False
        for i in range(9):
            if self[i] != other[i]:
                return False

    def __hash__(self):
        return hash((str(self.player_map), str(self.board_state)))
