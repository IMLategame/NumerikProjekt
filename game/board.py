from game.moves import Move

"""
    The main class in the game. Representation of the Board.
    We have the following indices:
    
     (2,0,0) ------------------------ (2,1,0) ------------------------ (2,2,0)
        |                                |                                |
        |                                |                                |
        |                                |                                |
        |      (1,0,0) -------------- (1,1,0) -------------- (1,2,0)      |
        |         |                      |                      |         |
        |         |                      |                      |         |
        |         |                      |                      |         |
        |         |      (0,0,0) ---- (0,1,0) ---- (0,2,0)      |         |
        |         |         |                         |         |         |
        |         |         |                         |         |         |
        |         |         |                         |         |         |
     (2,0,1) - (1,0,1) - (0,0,1)                   (0,2,1) - (1,2,1) - (2,2,1)
        |         |         |                         |         |         |
        |         |         |                         |         |         |
        |         |         |                         |         |         |
        |         |      (0,0,2) ---- (0,1,2) ---- (0,2,2)      |         |
        |         |                      |                      |         |
        |         |                      |                      |         |
        |         |                      |                      |         |
        |      (1,0,2) -------------- (1,1,2) -------------- (1,2,2)      |
        |                                |                                |
        |                                |                                |
        |                                |                                |
     (2,0,0) ------------------------ (2,1,0) ------------------------ (2,2,0)
        
     Well this was painfull ...
"""
class Board:
    def __init__(self, noP=(0, 0), p0=(1, 0), p1=(0, 1)):
        # possible states:
        self.player_map = {-1: noP, 0: p0, 1: p1}
        self.string_rep = {noP: "_", p0: "x", p1: "o"}

        # board is a matrix 3x3x3 of (0,0). 'None' in unnecessary matrix entries.
        # Board b -> b[r,x,y]
        # Access by self.board_state[ring][x][y] or self[ring, x, y] with inner ring = 0, middle ring = 1, outer ring = 2.
        # This is just the beginning board state
        self.board_state = [[[None if x == 1 and y == 1 else self.player_map[-1] for y in range(3)]
                             for x in range(3)] for ring in range(3)]

    # Where is no player?
    def get_empty_pos(self):
        return self.get_player_pos(-1)

    # Where is the specified player? No player is also an option (-1 / "noP")
    def get_player_pos(self, player):
        if player in ["a", "b", "no"]:
            player = {"a": 0, "b": 1, "no": -1}[player]
        assert player in [0, 1, -1]
        pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    #iteration over all legal r,x,y
                    if self[r, x, y] == self.player_map[player]:
                        pos.append((r, x, y))
        return pos

    # this is so that you can call
    # Board : b -> b[r,x,y]
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

    # this is so that you can call
    # Board : b -> b[r,x,y]
    def __getitem__(self, item):
        r, x, y = item
        possible = [0, 1, 2]
        assert x in possible
        assert y in possible
        assert r in possible
        assert x != 1 or y != 1
        return self.board_state[r][x][y]

    # This is so Board b -> str(b) looks fancy (and therefore also print(b))
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
                                                       self.string_rep[self[2, 2, 0]],
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

    def legal_moves(self, phase, player):
        assert phase in ["set", "move", "jump", "take"]
        assert player in [0, 1]
        moves = []
        if phase == "set":
            for x in range(3):
                for y in range(3):
                    if x == 1 and y == 1:
                        continue
                    for r in range(3):
                        if self[r, x, y] == self.player_map[-1]:
                            moves.append(Move("set", (r, x, y)))
        elif phase in ["move", "jump"]:
            start_vals = []
            for x in range(3):
                for y in range(3):
                    if x == 1 and y == 1:
                        continue
                    for r in range(3):
                        if self[r, x, y] == self.player_map[player]:
                            start_vals.append((r, x, y))
            end_vals = []
            for x in range(3):
                for y in range(3):
                    if x == 1 and y == 1:
                        continue
                    for r in range(3):
                        if self[r, x, y] == self.player_map[-1]:
                            end_vals.append((r, x, y))
            for start in start_vals:
                for end in end_vals:
                    move = Move("move", end, start)
                    if self.is_legal(move, phase, player):
                        moves.append(move)
        elif phase == "take":
            for x in range(3):
                for y in range(3):
                    if x == 1 and y == 1:
                        continue
                    for r in range(3):
                        if self[r, x, y] == self.player_map[1-player]:
                            moves.append(Move("take", (r, x, y)))
        return moves

    # Is the given move from the player in the phase legal given the baord state?
    def is_legal(self, move: Move, phase, player):
        # Is it even a move?
        assert phase in ["set", "move", "jump", "take"]
        assert player in [0, 1]
        assert move.end[1] in [0, 1, 2]
        assert move.end[2] in [0, 1, 2]
        assert move.end[0] in [0, 1, 2]
        assert move.end[1] != 1 or move.end[2] != 1
        if move.type == "move":
            assert move.start[0] in [0, 1, 2]
            assert move.start[1] in [0, 1, 2]
            assert move.start[2] in [0, 1, 2]
            assert move.start[1] != 1 or move.start[2] != 1

        enemy = 1 - player

        # Is the move correct for the phase?
        if phase == "set" and move.type != "set":
            return False
        if (phase == "move" or phase == "jump") and move.type != "move":
            return False
        if phase == "take" and move.type != "take":
            return False

        # Its a take move
        if move.type == "take":
            # Has to take an enemies piece
            if move.end not in self.get_player_pos(enemy):
                return False
        else:
            # Its a move of type set or move.
            # Therefore the endpoint has to be an empty position.
            if move.end not in self.get_empty_pos():
                return False
            #If its a move/jump move. Therefore the start point has to be your own piece (duh).
            if move.type == "move" and move.start not in self.get_player_pos(player):
                return False
            # Its in the move phase. No jumps allowed here.
            if phase == "move":
                start_r, start_x, start_y = move.start
                end_r, end_x, end_y = move.end

                #move in ring
                if start_r == end_r:
                    if abs(start_x-end_x) + abs(start_y-end_y) != 1:
                        return False

                # move between rings
                elif abs(start_r - end_r) != 1:
                    return False
                else:
                    if start_x != end_x or start_y != end_y:
                        return False
                    if start_x != 1 and start_y != 1:
                        return False
        return True

    # Apply the (legal) move to the current board
    def do(self, move: Move, player):
        if move.type == "set":
            self[move.end] = player
            return
        if move.type == "take":
            self[move.end] = -1
            return
        if move.type == "move":
            self[move.start] = -1
            self[move.end] = player

    # What are the current mulls?
    def get_mulls(self, player):
        mulls = []
        pieces = self.get_player_pos(player)
        for i in range(len(pieces)-2):
            for j in range(i+1, len(pieces)-1):
                for k in range(j+1, len(pieces)):
                    # all different sets of pieces of this player.
                    p1 = pieces[i]
                    p2 = pieces[j]
                    p3 = pieces[k]
                    # same ring
                    if p1[0] == p2[0] and p1[0] == p3[0]:
                        #same x
                        if p1[1] == p2[1] and p1[1] == p3[1]:
                            mulls.append((p1, p2, p3))
                        # same y
                        elif p1[2] == p2[2] and p1[2] == p3[2]:
                            mulls.append((p1, p2, p3))
                    # different ring, but same x and y
                    elif p1[1] == p2[1] and p1[1] == p3[1] and p1[2] == p2[2] and p1[2] == p3[2]:
                        mulls.append((p1, p2, p3))
        return mulls

    # Is the point part of a mull?
    def in_mull(self, player, point):
        mulls = self.get_mulls(player)
        for mull in mulls:
            if point in mull:
                return True
        return False

    def is_terminal(self, phase, player):
        assert phase in ["set", "move", "jump", "take"]
        assert player in [0, 1]
        if phase == "set":
            return False
        if len(self.get_player_pos(0)) <= 2 or len(self.get_player_pos(1)) <= 2:
            return True
        if len(self.legal_moves(phase, player)) == 0:
            return True
        return False
