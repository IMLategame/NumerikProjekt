
class Move:
    def __init__(self, move_type, end, start=None):
        # types of moves are move (a piece) (this also includes jumps), set (a piece) and take (an enemies piece)
        assert move_type in ["move", "set", "take"]
        self.type = move_type
        self.start = start
        self.end = end

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        if self.type != other.type:
            return False
        if self.end != other.end:
            return False
        if self.type == "move" and self.start != other.start:
            return False
        return True

    def __hash__(self):
        if self.type == "move":
            return hash((self.type, self.start, self.end))
        return hash((self.type, self.end))
