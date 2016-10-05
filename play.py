import board
import simple_dqn
import numpy as np
import random
import logging
import sys

def play_move(model, board, possible_moves, epsilon=0):
    """Play next move on a board"""
    r = None #move result
    inverted = False #True if board was inverted
    while r == None:
        #if it is Os turn - invert the board
        #so we can use dqn trained to play Xs
        if board.turn == -1:
            board.invert()
            inverted = True
        s = board.get_vec()
        #calculate q for all actions using dqn
        p = model.predict([s])[0]
        print(board)
        print(p)
        #make a random move or use dqn prediction
        if random.random() < epsilon:
            #a is an action(step) to be performed
            a = random.choice(list(possible_moves))
        else:
            #select best action based on model prediction
            a = np.argmax(p)
            if a not in possible_moves:
                a = random.choice(list(possible_moves))
        r = board.make_move(a)
        if inverted:
            board.invert()
            #invert reward too
            if r:
                r = (-r[0], r[1])
        if r == None:
            continue
        #remove action from possible moves set for this game
        possible_moves.remove(a)
        ss = board.get_vec()
        return (a,r,ss)

if __name__ == '__main__':

    #game number counter
    game_number=0
    wins=0

    game_result_map = {1: "X win!", -1: "O win", 0: "Draw"}

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    dqn = simple_dqn.getModel()

    while True:
        b = board.Board()
        #game state is pair of Reward,Terminal
        #where reward is 1 when X win, -1 when O win, and 0 otherwise
        #Terminal is true when the game is ended
        game_state = (0,False)
        #game step counter
        step = 0

        try:
            possible_moves = set([i for i in range(0,9)])
            #repeat until the game is ended
            while game_state[1] == False:
                step = step + 1
                r = None
                while r == None:
                    (a, r, ss) = play_move(dqn, b, possible_moves)
                if r[1]:
                    game_state=r
                    break
                print(b)
                print("Your move")
                m = int(input())
                r = b.make_move(m)
                print(b)
                if r[1]:
                    game_state=r
                    break

            log.info("Game %s ended in %s steps. %s", game_number, step, game_result_map[game_state[0]])
            log.info("\n%s",b)
            game_number = game_number + 1
            if game_state[0] == 1:
                wins = wins + 1
            log.info("Wining rate: %s", wins/game_number)

        except KeyboardInterrupt:
            sys.exit(1)

