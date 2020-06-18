
class Move:
    def __init__(self, type, end, start=None):
        # types of moves are move (a piece) (this also includes jumps), set (a piece) and take (an enemies piece)
        assert type in ["move", "set", "take"]
        self.type = type
        self.start = start
        self.end = end
