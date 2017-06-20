import board
import numpy as np
import unittest

class TestBoard(unittest.TestCase):

    def test_analyze(self):
        test_tab = [[None,(0,False)],
                    [[" "," "," "," "," "," "," "," "," "],(0,False)],
                    [["X","X","X"," "," "," "," "," "," "],(1,True)],
                    [["X","X","X","X","X","X","X","X","X"],(1,True)],
                    [["X","O","O","X"," "," ","X"," "," "],(1,True)],
                    [["O","O","O"," "," "," "," "," "," "],(-1,True)],
                    [["O","X","O"," ","O"," ","O"," "," "],(-1,True)],
                    [["O","X","O"," ","O"," ","X"," "," "],(0,False)]]
        for test in test_tab:
            b = board.Board(test[0])
            r = b.analyze_state()
            print(b)
            print("Reward: {}, terminal: {}".format(r[0],r[1]))
            self.assertEqual(r,test[1])

    def test_get_vec(self):
        test_tab = [[None,[np.float32(0)]*9],
                    [["X","O","X"," "," "," "," "," "," "],
                    [np.float32(1),np.float32(-1),np.float32(1)] + [np.float32(0)]*6]]
        for test in test_tab:
            self.assertEqual(board.Board(test[0]).get_vec(),test[1])

    def test_make_move(self):
        test_tab = [[None,1,0,(0,False),["X"," "," "," "," "," "," "," "," "]],
                    [None,-1,0,(0,False),["O"," "," "," "," "," "," "," "," "]],
                    [["X","X","X","X","X","X","X","X","X"],1,3,None,["X","X","X","X","X","X","X","X","X"]],
                    [["X","X","X","X","X","X","X","X","X"],-1,5,None,["X","X","X","X","X","X","X","X","X"]],
                    [["X","X"," "," "," "," "," "," "," "],1,2,(1,True),["X","X","X"," "," "," "," "," "," "]],
                    [["O","O"," "," "," "," "," "," "," "],-1,2,(-1,True),["O","O","O"," "," "," "," "," "," "]]]
        for test in test_tab:
            b = board.Board(test[0],test[1])
            print("Pre move")
            print(b)
            r = b.make_move(test[2])
            print("Post move ({})".format(test[2]))
            print(b)
            if r:
                print("Reward: {}, terminal: {}".format(r[0],r[1]))
            else:
                print("Invalid move")

            self.assertEqual(r,test[3],"Result (r,term) does not match")
            self.assertEqual(b.data,test[4],"Board state does not match expected")


if __name__ == '__main__':
    unittest.main()
