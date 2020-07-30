from copy import deepcopy
from network_backend.reinforcement_learning.encodings import EncodingI
from network_backend.reinforcement_learning.goalFunctions import GoalFunctionI


class ReplayMem:
    def __init__(self, capacity, batch_size, Q_fctn, gamma, encode: EncodingI, goal_value_function: GoalFunctionI):
        self.N = capacity
        self.Q_fctn = Q_fctn
        self.gamma = gamma
        self.encode = encode
        self.goal = goal_value_function
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
            (self.encode(p[2], p[0], p[1], p[6]), self.goal(self.Q_fctn, self.encode, self.gamma, p[0], p[1], p[2],
                                                            p[3], p[4], p[5], p[6])) for p in self.mem]

    def get_v_data(self):
        data_list = []
        # simulate outcome:
        for prev, phase_p, a, r, post, phase_p, playerID in self.mem:
            intermed = deepcopy(prev)
            intermed.do(move=a, player=playerID)
            data_list.append((self.encode(a, intermed, None, playerID), self.goal(self.Q_fctn, self.encode, self.gamma, prev, phase_p, a, r, post, phase_p, playerID)))
        return data_list


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

    def get_current_mem(self):
        return self.mem


class DataMemory:
    def __init__(self, num_games):
        self.num_games = num_games
        self.data = []

    def add_game_data(self, dataset):
        if len(self.data) >= self.num_games:
            self.data = self.data[1:]
        self.data.append(dataset)

    def get_data(self):
        dataset = []
        for game_data in self.data:
            dataset += game_data
        return dataset

    def __len__(self):
        size = 0
        for game_data in self.data:
            size += len(game_data)
        return size
