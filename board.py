
class Board(object):
    """Tic tac toe board representation class"""

    def __init__(self, data=None, turn=None):
        """Initializes new tictactoe Board
        :param data: 9 element list. 0 element is top left corner, 8 is bottom right. Use X and O characters and space for empty squares
        if data is None, empty board (filed with None) is created
        :param turn: 1 is for X's turn -1 for O's turn"""
        if not data:
            self.data = [' ']*9
        elif len(data) != 9:
            raise ValueError("Board size should be 9")
        else:
            self.data = data
        if turn:
            self.turn = turn
        else:
            turn = len([c for c in self.data if c != ' '])%2
            if turn == 0:
                self.turn = 1
            else:
                self.turn = -1

    def analyze_state(self):
        """Analyze board data and return
            (WhoWins, EndOfGame) tuple:
            * (1,True) if X win
            * (-1,True) if O win
            * (0,True) if draw
            * (0, False) if board is not full"""

        line_mask = [[0,1,2],[3,4,5],[6,7,8], #rows
                     [0,3,6],[1,4,7],[2,5,8], #columns
                     [0,4,8],[2,4,6]]         #diagonals

        for mask in line_mask:
            line = "".join([self.data[i] for i in mask])
            if line == "XXX":
                return (1, True)
            if line == "OOO":
                return (-1, True)

        for cell in self.data:
            if cell == ' ':
                return (0, False)

        return (0, True)

    def get_vec(self, ):
        "Float vector board representation for ML applications"
        float_map = {"X": 1,
                     "O": -1,
                     " ": 0}
        return [float_map[c] for c in self.data]

    def invert(self):
        "Invert board to use DQN trained to win with X for playng for O"
        for i in range(0,9):
            if self.data[i] == "X":
                self.data[i] = "O"
            elif self.data[i] == "O":
                self.data[i] = "X"
        self.turn = -self.turn

    def make_move(self, action):
        """Makes move on current board (modifies its state). Returns reward and terminal status, or None if move makes no sense.
        :param action: int in range(0,9) - board cell to make move on"""
        if self.data[action] != ' ':
            return None
        turn_map = {1:"X", -1:"O"}
        self.data[action] = turn_map[self.turn]
        self.turn = -self.turn

        return self.analyze_state()

    def __str__(self):
        "String representation for debugging output"
        s = "|".join([c for c in self.data[0:3]]) + "\n"
        s += "-----\n"
        s += "|".join([c for c in self.data[3:6]]) + "\n"
        s += "-----\n"
        s += "|".join([c for c in self.data[6:9]])
        return s
