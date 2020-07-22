from copy import deepcopy
import random

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

    def get_minibatch(self):
        assert len(self.dataset) > 0, "No data in memory."
        shuffeled_data = []
        while len(shuffeled_data) < self.batch_size:
            shuffeled_data += list(self.dataset)
        shuffeled_data = random.shuffle(shuffeled_data)
        batch = shuffeled_data[:self.batch_size]
        batch_x = [p[0:3]+[p[6]] for p in batch]
        batch_y = [p[3] if p[0].isTerminal(p[2], p[6]) else p[3] + self.gamma * self.Q_fctn(p[0], p[1], p[2], p[6])
                   for p in batch]
        return batch_x, batch_y
