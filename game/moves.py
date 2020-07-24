
class Move:
    def __init__(self, move_type, end, start=None):
        # types of moves are move (a piece) (this also includes jumps), set (a piece) and take (an enemies piece)
        assert move_type in ["move", "set", "take"]
        self.type = move_type
        self.start = start
        self.end = end
