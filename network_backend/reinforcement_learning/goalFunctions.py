from network_backend.reinforcement_learning.encodings import EncodingI
from NineMenMorris.moves import Move


class GoalFunctionI:
    def call(self, net, encoding: EncodingI, gamma, prev, phase_prev, a: Move, r, post, phase_post, turn_player):
        raise NotImplementedError()

    def __call__(self, net, encoding: EncodingI, gamma, prev, phase_prev, a: Move, r, post, phase_post, turn_player):
        return self.call(net, encoding, gamma, prev, phase_prev, a, r, post, phase_post, turn_player)


class QGoal(GoalFunctionI):
    def call(self, net, encoding: EncodingI, gamma, prev, phase_prev, a, r, post, phase_post, turn_player):
        if post.is_terminal(phase=phase_post, player=turn_player):
            return r
        max_Q = -2**62
        legal = post.legal_moves(phase_post, turn_player)
        if len(legal) == 0:
            return r
        for move in legal:
            q_val = net(encoding(move, post, phase_post, turn_player))[0][0]
            if q_val > max_Q:
                max_Q = q_val
        return r + gamma * max_Q


class VGoal(GoalFunctionI):
    def call(self, net, encoding: EncodingI, gamma, prev, phase_prev, a: Move, r, post, phase_post, turn_player):
        if post.is_terminal(phase_post, turn_player):
            return r
        # assumption: the action a is the best possible, given the current state of the net (/the state value function)
        return r + gamma * net(encoding(None, post, phase_post, turn_player))[0][0]
