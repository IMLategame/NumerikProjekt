from game.board import Board
from game.moves import Move
import re

"""
    Interface for Players.
    Subclass this to create a new player (something that plays the game :) )
"""
class PlayerI:
    def __init__(self, playerID=0):
        # Player has an ID that is either 0 or 1
        if playerID in ["a", "b"]:
            playerID = {"a":0, "b":1}[playerID]
        assert playerID in [0, 1]
        self.playerID = playerID

    # This is the method to implement in 'real' player-classes. It takes a board and the game-phase and should in the implementation return a legal move.
    def get_move(self, phase, board: Board):
        assert phase in ["set", "move", "jump", "take"]
        raise NotImplementedError()

    # What the player does when it wins. Can change that in players.
    def win(self):
        print("Player {}: I won :)".format(self.playerID))

# Just parsing a string with some (>= 3) numbers in it to the numbers (only the first 3)
def parse_point(string):
    print("point stuff:")
    print([int(s) for s in re.findall(r'\d+', string)])
    r, x, y = [int(s) for s in re.findall(r'\d+', string)][:3]
    return r, x, y

"""
    The first and most simple implementation of a player. Just asks the user what to do on the command line.
"""
class CmdPlayer(PlayerI):
    def __init__(self, playerID=0):
        # Call this first in your implementations of the player also.
        super().__init__(playerID)

    # The function we have this class for
    def getMove(self, phase, board: Board):
        print("Its your turn. You are player {} \n The phase is {}".format(board.string_rep[board.player_map[self.playerID]], phase))
        print(board)
        # continue asking until we get a reasonable answer
        while True:
            print("Whats your move? Possibilities are 'mv <start>; <end>', 'st <point>' or 'tk <point>'.")
            move_string = input("->")
            # is the input possibly a command?
            if move_string[:2] not in ["mv", "st", "tk"]:
                continue

            # its a move command
            if move_string[:2] == "mv":
                # we have 2 points on the board with a ; somewhere in between!
                points = move_string[2:].split(";")
                start_p = parse_point(points[0])
                end_p = parse_point(points[1])
                move = Move("move", end_p, start_p)
            # its a set command
            elif move_string[:2] == "st":
                end_p = parse_point(move_string[2:])
                move = Move("set", end_p)
            # its a take command
            else:
                end_p = parse_point(move_string[2:])
                move = Move("take", end_p)
            # is this command even legal?
            if board.is_legal(move, phase, self.playerID):
                return move
