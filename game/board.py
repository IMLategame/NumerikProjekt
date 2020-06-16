
class board:
    def __init__(self):
        # possible states:
        self.empty = (0, 0)
        self.p0 = (1, 0)
        self.p1 = (0, 1)

        # board is a matrix 3x3x3 of (0,0). Non in unnecessary matrix entries.
        # Access by [ring, x, y] with inner ring = 0, middle ring = 1, outer ring = 2.
        self.board_state = [[[None if x == 1 and y == 1 else self.empty for y in range(3)] for x in range(3)] for ring in range(3)]

    def get_empty_pos(self):
        pos = []
        for x in range(3):
            for y in range(3):
                if x == 2 and y == 2:
                    continue
                for r in range(3):
                    if self.board_state[r, x, y] == self.empty:
                        pos.append((r, x, y))
        return pos

    def get_player_pos(self, player):
        assert player in [0,1]
        player_map = {0 : self.p0, 1 : self.p1}
        pos = []
        for x in range(3):
            for y in range(3):
                if x == 2 and y == 2:
                    continue
                for r in range(3):
                    if self.board_state[r, x, y] == player_map[player]:
                        pos.append((r, x, y))
        return pos
