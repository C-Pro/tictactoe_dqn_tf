class Board(object):
    def __init__(self, data):
        self.data = data

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
            line = "".join([self.data[i] or " " for i in mask])
            if line == "XXX":
                return (1, True)
            if line == "OOO":
                return (-1, True)

        for cell in self.data:
            if cell == None:
                return (0, False)

        return (0, True)

    def __str__(self):
        s = "|".join([c or " " for c in self.data[0:3]]) + "\n"
        s += "-----\n"
        s += "|".join([c or " " for c in self.data[3:6]]) + "\n"
        s += "-----\n"
        s += "|".join([c or " " for c in self.data[6:9]])
        return s


import unittest

class TestBoard(unittest.TestCase):

    test_tab = [[[None,None,None,None,None,None,None,None,None],(0,False)],
                [["X","X","X",None,None,None,None,None,None],(1,True)],
                [["X","X","X","X","X","X","X","X","X"],(1,True)],
                [["X","O","O","X",None,None,"X",None,None],(1,True)],
                [["O","O","O",None,None,None,None,None,None],(-1,True)]]

    def test_analyze(self):
        for test in self.test_tab:
            b = Board(test[0])
            r = b.analyze_state()
            print(b)
            print(r)
            self.assertEqual(r,test[1])

if __name__ == '__main__':
    unittest.main()

