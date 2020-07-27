import numpy as np


class EncodingI:
    @classmethod
    def call(cls, move, board, phase, playerID):
        raise NotImplementedError()

    def __call__(self, move, board, phase, playerID):
        return self.call(move, board, phase, playerID)


class QEncoding(EncodingI):
    @classmethod
    def call(cls, move, board, phase, playerID):
        assert phase in ["set", "move", "jump", "take"]
        phase2idx = {"set": 0, "move": 1, "jump": 2, "take": 3}
        moveType2idx = {"set": 0, "move": 1, "take": 2}
        phase_enc = [0.0, 0.0, 0.0, 0.0]
        phase_enc[phase2idx[phase]] = 1.0
        my_pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if board[r, x, y] == board.player_map[playerID]:
                        my_pos.append(1.0)
                    else:
                        my_pos.append(0.0)
        enemy_pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if board[r, x, y] == board.player_map[1 - playerID]:
                        enemy_pos.append(1.0)
                    else:
                        enemy_pos.append(0.0)
        move_type_enc = [0.0, 0.0, 0.0]
        move_type_enc[moveType2idx[move.type]] = 1.0
        move_start = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if (r, x, y) == move.start:
                        move_start.append(1.0)
                    else:
                        move_start.append(0.0)
        move_end = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if (r, x, y) == move.end:
                        move_end.append(1.0)
                    else:
                        move_end.append(0.0)
        assert len(phase_enc) == 4
        assert len(my_pos) == 24
        assert len(enemy_pos) == 24
        assert len(move_type_enc) == 3
        assert len(move_start) == 24
        assert len(move_end) == 24
        enc = phase_enc + my_pos + enemy_pos + move_type_enc + move_start + move_end
        return np.array(enc)


class VEncoding(EncodingI):
    @classmethod
    def call(cls, move, board, phase, playerID):
        assert phase in ["set", "move", "jump", "take"]
        phase2idx = {"set": 0, "move": 1, "jump": 2, "take": 3}
        phase_enc = [0.0, 0.0, 0.0, 0.0]
        phase_enc[phase2idx[phase]] = 1.0
        my_pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if board[r, x, y] == board.player_map[playerID]:
                        my_pos.append(1.0)
                    else:
                        my_pos.append(0.0)
        enemy_pos = []
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                for r in range(3):
                    if board[r, x, y] == board.player_map[1 - playerID]:
                        enemy_pos.append(1.0)
                    else:
                        enemy_pos.append(0.0)
        enc = phase_enc + my_pos + enemy_pos
        return np.array(enc)


class TTTQEncoding(EncodingI):
    @classmethod
    def call(cls, move, board, phase, playerID):
        my_pos = [0.0 for _ in range(9)]
        for pos in board.get_player_pos(playerID):
            my_pos[pos] = 1.0
        enemy_pos = [0.0 for _ in range(9)]
        for pos in board.get_player_pos(1-playerID):
            enemy_pos[pos] = 1.0
        move_pos = [0.0 for _ in range(9)]
        move_pos[move] = 1.0
        encoding = my_pos + enemy_pos + move_pos
        return np.array(encoding)
