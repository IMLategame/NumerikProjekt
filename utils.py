from copy import deepcopy
import random
import numpy as np


# encode all information one-hot.
def encode(move, board, phase, playerID):
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


class ReplayMem:
    def __init__(self, capacity, batch_size, Q_fctn, gamma):
        self.N = capacity
        self.batch_size = batch_size
        self.Q_fctn = Q_fctn
        self.gamma = gamma
        self.mem = []

    def add(self, state_prev, phase_prev, action, reward, state_post, phase_post, turn_player=0):
        if len(self.mem) >= self.N:
            self.mem = self.mem[1:]
        prev = deepcopy(state_prev)
        a = action
        r = reward
        post = deepcopy(state_post)
        self.mem.append([prev, phase_prev, a, r, post, phase_post, turn_player])

    def get_data(self):
        return [
            (encode(p[2], p[0], p[1], p[6]), p[3]) if p[4].is_terminal(p[5], p[6]) else
            (encode(p[2], p[0], p[1], p[6]), p[3] + self.gamma * self.Q_fctn(encode(p[2], p[4], p[5], p[6]))[0][0]) for
            p in self.mem]

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, item):
        assert item < len(self)
        return self.mem[item]

    def __str__(self):
        string = ""
        for data in self.mem:
            string += str(data) + "\n"
        return string
