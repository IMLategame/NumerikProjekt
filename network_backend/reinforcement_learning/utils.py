from copy import deepcopy
import random


class ReplayMem:
    def __init__(self, capacity, batch_size, Q_fctn, gamma, encode, goal_value_function):
        self.N = capacity
        self.batch_size = batch_size
        self.Q_fctn = Q_fctn
        self.gamma = gamma
        self.encode = encode

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
            (self.encode(p[2], p[0], p[1], p[6]), p[3]) if p[4].is_terminal(p[5], p[6]) else
            (self.encode(p[2], p[0], p[1], p[6]), p[3] + self.gamma * self.Q_fctn(
                self.encode(p[2], p[4], p[5], p[6]))[0][0]) for p in self.mem]

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
