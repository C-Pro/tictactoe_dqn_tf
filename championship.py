#!/usr/bin/env python3

import board
import simple_dqn
import play
from glob import glob
import logging

#how many rounds to play
ROUNDS = 1000
#probability of random move
EPSILON = 0.15

if __name__ == '__main__':

    game_result_map = {1: "X win!", -1: "O win", 0: "Draw"}

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    models = [glob("X*.model.index"),glob("O*.model.index")]
    scores = [[0]*len(models[0]),[0]*len(models[1])]
    dqns = [[simple_dqn.getModel(models[0][i]) for i in range(len(models[0]))],
            [simple_dqn.getModel(models[1][i]) for i in range(len(models[1]))]]

    for round in range(ROUNDS):
        for xn in range(len(models[0])):
            for on in range(len(models[1])):
                dqn = [dqns[0][xn], dqns[1][on]]

                b = board.Board()
                #game state is pair of Reward,Terminal
                #where reward is 1 when X win, -1 when O win, and 0 otherwise
                #Terminal is true when the game is ended
                game_state = (0,False)
                #game step counter
                step = 0
                possible_moves = set([i for i in range(0,9)])
                #repeat until the game is ended
                while game_state[1] == False:
                    side = step % 2
                    #print(b)
                    #if print_eval:
                    #    p = dqn[side][0].predict([b.get_vec()])[0]
                    #    print_weights(p)
                    r = None
                    (a, r, ss, possible_moves) = play.play_move(dqn[side][0], b, possible_moves, EPSILON)
                    game_state = r
                    step = step + 1

                log.info("%s vs %s: %s",
                         models[0][xn],
                         models[1][on],
                         game_result_map[game_state[0]])

                scores[0][xn] += game_state[0]
                scores[1][on] -= game_state[0]

    log.info("X models ordered by their scores:\n%s",
             sorted(zip(models[0], scores[0]), key=lambda z:z[1]))
    log.info("O models ordered by their scores:\n%s",
             sorted(zip(models[1], scores[1]), key=lambda z:z[1]))
