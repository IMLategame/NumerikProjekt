
class Move:
    def __init__(self, type, end, start=None):
        # types of moves are move (a piece) and set (a piece)
        assert type in ["move", "set"]
        self.type = type
        self.start = start
        self.end = end
