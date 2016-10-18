#!/usr/bin/env python3

import board
import simple_dqn
import numpy as np
from termcolor import colored
import random
import logging
import sys

def print_weights(q):
    '''Pretty-print Q values on board'''
    maxval = np.max(q)
    for l in range(3):
        s =  "|".join(["{: 04.3f}".format(x) if x<maxval else \
                colored("{: 04.3f}".format(x),"red") for x in q[l*3:l*3+3]])
        print(s)

def play_move(model, board, possible_moves, epsilon=0):
    """Play next move on a board"""
    r = None #move result
    while r == None:
        s = board.get_vec()

        #make a random move or use dqn prediction
        if random.random() < epsilon:
            #a is an action(step) to be performed
            a = random.choice(list(possible_moves))
        else:
            #calculate q for all actions using dqn
            p = model.predict(simple_dqn.reshape(s))[0]
            #select best action based on model prediction
            p = [x if i in possible_moves else -1.0 for (i,x) in enumerate(p)]
            a = np.argmax(p)
            if a not in possible_moves:
                a = random.choice(list(possible_moves))
        r = board.make_move(a)
        if r == None:
            continue
        #remove action from possible moves set for this game
        possible_moves.remove(a)
        ss = board.get_vec()
        return (a,r,ss, possible_moves)

if __name__ == '__main__':

    #game number counter
    game_number=0
    wins=0

    computer_plays = [0]
    if "-xo" in sys.argv:
        computer_plays = [0,1]
    elif "-o" in sys.argv:
        computer_plays = [1]

    print_eval = "--eval" in sys.argv

    game_result_map = {1: "X win!", -1: "O win", 0: "Draw"}

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    dqn = [simple_dqn.getModel("x.model"), simple_dqn.getModel("o.model")]

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
                side = step % 2
                print(b)
                if print_eval:
                    p = dqn[side][0].predict(simple_dqn.reshape(b.get_vec()))[0]
                    print_weights(p)
                r = None
                if side in computer_plays:
                    print("My move:")
                    (a, r, ss, possible_moves) = play_move(dqn[side][0], b, possible_moves)
                else:
                    while r ==None:
                        print("Your move:")
                        m = int(input())
                        r = b.make_move(m)
                        if r == None:
                            print("Wrong move!")
                    possible_moves.remove(m)
                game_state = r
                step = step + 1

            log.info("Game %s ended in %s steps. %s", game_number, step, game_result_map[game_state[0]])
            log.info("\n%s",b)
            game_number = game_number + 1
            if game_state[0] == 1:
                wins = wins + 1
            log.info("Wining rate: %s", wins/game_number)

        except KeyboardInterrupt:
            sys.exit(1)

