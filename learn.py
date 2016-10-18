#!/usr/bin/env python3

import board
import simple_dqn
import play
import numpy as np
import random
import logging
import sys

#game number counter
game_number=0
wins=0
losses=0

#start with random move generation, and slowly decrease randomness
epsilon=0.99
epsilon_min=0.1
epsilon_step=(epsilon-epsilon_min)/500000
#gamma is future reward discount in Bellman equation
gamma=0.9

batch_size = 40000
min_batch_size = 500

game_result_map = {1: "X win!", -1: "O win", 0: "Draw"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
filenames = ["x.model", "o.model"]
dqnX = simple_dqn.getModel(filenames[0])
dqnO = simple_dqn.getModel(filenames[1])

replay =[[],[]]
dqn = [dqnX, dqnO]

def sample_replay(replay, gamma, batch_size):
    '''Get a "random" batch of replay steps
    prefer steps closest to endgame, slowly decreasing this "preference"
    using gamma parameter'''
    res = []
    cnt = 0
    while (len(res) < batch_size) and (cnt < len(replay)):
        sample = replay[int(random.random() * len(replay))]
        if gamma > 0.8 and sample["terminal"]:
            res.append(sample)
        elif random.random()+gamma/2 < pow((((sample["step"]+1)/9)/gamma),3):
            res.append(sample)
        cnt = cnt + 1

    return res


while True:

    b = board.Board()
    # game state is pair of Reward,Terminal
    # where reward is 1 when X win, -1 when O win, and 0 otherwise
    # Terminal is true when the game is ended
    r = (0, False)
    # game step counter
    step = 0

    try:
        log.debug("Starting game %d",game_number)
        possible_moves = set([i for i in range(0,9)])
        prev_state = None
        prev_board_vec = None
        #repeat until the game is ended
        while r[1] == False:
            side = step % 2
            s = b.get_vec()
            #play next move
            (a, r, ss, possible_moves) = play.play_move(dqn[side][0], b, possible_moves, epsilon)
            #save previous move and current reward to another side's replay
            if prev_state:
                replay[1-side].append({"state0": prev_board_vec,
                                      "state1": ss,
                                      "step": step,
                                      "action": prev_state[0],
                                      #invert revard for Os
                                      "reward": r[0] if (1-side)==0 else -r[0],
                                      "terminal": r[1]})
            #for terminal moves save current move to this side replay
            if r[1]:
                replay[side].append({"state0": s,
                                     "state1": ss,
                                     "step": step,
                                     "action": a,
                                     #invert revard for Os
                                     "reward": r[0] if side==0 else -r[0],
                                     "terminal": r[1]})
            prev_state = (a, r, ss)
            prev_board_vec = s
            step = step + 1

        if game_number%1000 == 0:
            log.info("Sample endgame position for game %s:\n%s\n%s",game_number,b,game_result_map[r[0]])
        game_number = game_number + 1
        if r[0] == 1:
            wins = wins + 1
        if r[0] == -1:
            losses = losses + 1

        if game_number % batch_size == 0:
            log.info("After %s games played", game_number)
            win_rate = wins/batch_size
            loss_rate = losses/batch_size
            if win_rate == 0 and loss_rate == 0:
                log.info("Only draws in batch. Exiting.")
                sys.exit()
            log.info("  Wining rate for last %d games: %s", batch_size, win_rate)
            log.info("  Losing rate for last %d games: %s", batch_size, loss_rate)
            log.info("Sample weights for empty board:")
            play.print_weights(dqn[0][0].predict(simple_dqn.reshape([0,0,0,0,0,0,0,0,0]))[0])
            for side in range(2):
                batch = sample_replay(replay[side], gamma, batch_size)
                # Use Bellman equation and model prediction to backpropagate reward
                y = simple_dqn.calc_target(dqn[side][0], batch, gamma)
                simple_dqn.train(*dqn[side], batch, y, game_number, filenames[side], int(10*(batch_size/len(batch))))
            wins = losses = 0
            # Decrease epsilon
            if epsilon > epsilon_min:
                epsilon = epsilon - epsilon_step*batch_size
            # Reduce batch size
            if batch_size > min_batch_size:
                batch_size = int(batch_size*0.95)
        if game_number % 200001 == 0:
            random.shuffle(replay[0])
            replay[0] = replay[0][:100000]
            random.shuffle(replay[1])
            replay[1] = replay[1][:100000]


    except KeyboardInterrupt:
        sys.exit(1)

