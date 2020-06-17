from game.moves import Move

class Board:
    def __init__(self, noP=(0, 0), p0=(1, 0), p1=(0, 1)):
        # possible states:
        self.empty = (0, 0)
        self.player_map = {-1: noP, 0: p0, 1: p1}
        self.string_rep = {noP: "_", p0: "x", p1: "o"}

        # board is a matrix 3x3x3 of (0,0). Non in unnecessary matrix entries.
        # Access by self.board_state[ring][x][y] or self[ring, x, y] with inner ring = 0, middle ring = 1, outer ring = 2.
        self.board_state = [[[None if x == 1 and y == 1 else self.empty for y in range(3)] for x in range(3)] for ring
                            in range(3)]

    def get_empty_pos(self):
        return self.get_player_pos(-1)

    def get_player_pos(self, player):
        if player in ["a", "b", "no"]:
            player = {"a": 0, "b": 1, "no": -1}[player]
        assert player in [0, 1, -1]
        pos = []
        for x in range(3):
            for y in range(3):
                if x == 2 and y == 2:
                    continue
                for r in range(3):
                    if self.board_state[r, x, y] == self.player_map[player]:
                        pos.append((r, x, y))
        return pos

    def __setitem__(self, key, value):
        r, x, y = key
        possible = [0, 1, 2]
        assert x in possible
        assert y in possible
        assert r in possible
        assert x != 1 or y != 1
        if value in ["a", "b", "no"]:
            value = {"a": 0, "b": 1, "no": -1}[value]
        assert value in [0, 1, -1]
        self.board_state[r][x][y] = self.player_map[value]

    def __getitem__(self, item):
        r, x, y = item
        possible = [0, 1, 2]
        assert x in possible
        assert y in possible
        assert r in possible
        assert x != 1 or y != 1
        return self.board_state[r][x][y]

    def __str__(self):
        string = "\n\t{}---------{}---------{}" \
                 "\n\t|         |         |" \
                 "\n\t|  {}------{}------{}  |" \
                 "\n\t|  |      |      |  |" \
                 "\n\t|  |  {}---{}---{}  |  |" \
                 "\n\t|  |  |       |  |  |" \
                 "\n\t{}--{}--{}       {}--{}--{}" \
                 "\n\t|  |  |       |  |  |" \
                 "\n\t|  |  {}---{}---{}  |  |" \
                 "\n\t|  |      |      |  |" \
                 "\n\t|  {}------{}------{}  |" \
                 "\n\t|         |         |" \
                 "\n\t{}---------{}---------{}".format(self.string_rep[self[2, 0, 0]], self.string_rep[self[2, 1, 0]],
                                                       self.string_rep[self[1, 2, 0]],
                                                       self.string_rep[self[1, 0, 0]], self.string_rep[self[1, 1, 0]],
                                                       self.string_rep[self[1, 2, 0]],
                                                       self.string_rep[self[0, 0, 0]], self.string_rep[self[0, 1, 0]],
                                                       self.string_rep[self[0, 2, 0]],
                                                       self.string_rep[self[2, 0, 1]], self.string_rep[self[1, 0, 1]],
                                                       self.string_rep[self[0, 0, 1]],
                                                       self.string_rep[self[0, 2, 1]], self.string_rep[self[1, 2, 1]],
                                                       self.string_rep[self[2, 2, 1]],
                                                       self.string_rep[self[0, 0, 2]], self.string_rep[self[0, 1, 2]],
                                                       self.string_rep[self[0, 2, 2]],
                                                       self.string_rep[self[1, 0, 2]], self.string_rep[self[1, 1, 2]],
                                                       self.string_rep[self[1, 2, 2]],
                                                       self.string_rep[self[2, 0, 2]], self.string_rep[self[2, 1, 2]],
                                                       self.string_rep[self[2, 2, 2]])
        return string

    def is_legal(self, move: Move, phase, player):
        assert phase in ["set", "move", "jump", "take"]
        assert player in [0, 1]
        enemy = 1 - player
        if phase == "set" and move.type != "set":
            return False
        if (phase == "move" or phase == "jump") and move.type != "move":
            return False
        if phase == "take" and move.type != "take":
            return False
        if move.type == "take":
            if move.end not in self.get_player_pos(enemy):
                return False
        else:
            if move.end not in self.get_empty_pos():
                return False
            if move.type == "move" and move.start not in self.get_player_pos(player):
                return False
            if phase == "move":
                start_r, start_x, start_y = move.start
                end_r, end_x, end_y = move.end

                #move in ring
                if start_r == end_r:
                    if abs(start_x-end_x) + abs(start_y-end_y) > 1:
                        return False

                # move between rings
                elif abs(start_r - end_r) > 1:
                    return False
                else:
                    if start_x != end_x or start_y != end_y:
                        return False
        return True
