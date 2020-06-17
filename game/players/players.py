from game.board import Board

"""Interface for Players"""
class PlayerI:
    def __init__(self, playerID=0):
        assert playerID in [0,1]
        self.playerID = playerID

    def setPlayerID(self, ID):
        assert ID in [0,1]
        self.playerID = ID

    def getMove(self, phase, board: Board):
        assert phase in ["set", "move", "jump", "take"]
        raise NotImplementedError()

class CmdPlayer(PlayerI):
