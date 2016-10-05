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
epsilon_step=(epsilon-epsilon_min)/5000
#gamma is future reward discount in Bellman equation
gamma=0.8

game_result_map = {1: "X win!", -1: "O win", 0: "Draw"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
dqn = simple_dqn.getModel()

def invert_vector(v):
    inv_map = {-1: 1, 1: -1}
    return [inv_map[x] for x in v]

def invert_replay(replay):
    """Append inverted replay to original one"""
    inverted_rep = []
    for step in replay:
        inverted_rep.append({"state0": invert_vector(replay["state0"]),
                             "state1": invert_vector(replay["state1"]),
                             "action": replay["a"],
                             "reward": -replay["reward"]
                             })
    return inverted_rep


global_replay = []
global_y = []

while True:
    replay = []

    b = board.Board()
    #game state is pair of Reward,Terminal
    #where reward is 1 when X win, -1 when O win, and 0 otherwise
    #Terminal is true when the game is ended
    r = (0, False)
    #game step counter
    step = 0

    try:
        log.info("Starting game %d",game_number)
        possible_moves = set([i for i in range(0,9)])
        #repeat until the game is ended
        while r[1] == False:
            step = step + 1
            s = b.get_vec()
            #play next move as X
            (a, r, ss) = play.play_move(dqn, b, possible_moves, epsilon)

            if r[1] == False:
                #play next move as O
                (aO, r, ss) = play.play_move(dqn, b, possible_moves, epsilon)
            #save move to replay
            replay.append({"state0": s,
                           "state1": ss,
                           "action": a,
                           "reward": r[0],
                           "terminal": r[1]})

        log.info("Game %s ended in %s steps. %s", game_number, step, game_result_map[r[0]])
        game_number = game_number + 1
        if r[0] == 1:
            wins = wins + 1
        if r[0] == -1:
            losses = losses + 1
        log.info("Wining rate: %s", wins/game_number)
        log.info("Losing rate: %s", losses/game_number)

        if random.random() < epsilon + 10:
            y = simple_dqn.calc_target2(dqn, replay, gamma)
        else:
            random.shuffle(replay)
            # Use Bellman equation and model prediction to backpropagate reward
            y = simple_dqn.calc_target(dqn, replay, gamma)
            # Train model on replay

        global_replay = global_replay + replay
        global_y = global_y + y
        if game_number % 1000 == 0:
            simple_dqn.train(dqn, global_replay, global_y, game_number)
            global_replay = []
            global_y = []

        # Decrease epsilon with every step
        if epsilon > epsilon_min:
            epsilon = epsilon - epsilon_step
    except KeyboardInterrupt:
        sys.exit(1)

