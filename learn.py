import board
import simple_dqn
import numpy as np
import random
import logging
import sys

#game number counter
game_number=0
wins=0

#start with random move generation, and slowly decrease randomness
epsilon=0.99
epsilon_min=0.01
epsilon_step=(epsilon-epsilon_min)/100
#gamma is future reward discount in Bellman equation
gamma=0.8

game_result_map = {1: "X win!", -1: "O win", 0: "Draw"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
dqn = simple_dqn.getModel()

while True:
    replay = []
    b = board.Board()
    #game state is pair of Reward,Terminal
    #where reward is 1 when X win, -1 when O win, and 0 otherwise
    #Terminal is true when the game is ended
    game_state = (0,False)
    #game step counter
    step = 0

    try:
        log.info("Starting game #{}".format(game_number))
        possible_moves = set([i for i in range(0,9)])
        #repeat until the game is ended
        while game_state[1] == False:
            step = step + 1
            #s is initial state (before action)
            s = b.get_vec()
            #calculate q for all actions using dqn
            p = dqn.predict([s])[0]
            #make a random move or use dqn prediction
            if random.random() < epsilon:
                #a is an action(step) to be performed
                a = random.choice(list(possible_moves))
            else:
                #select best action based on model prediction
                a = np.argmax(p)
            #make chosen move
            r = b.make_move(a)
            #if move not possible - choose another
            if r == None:
                continue
            #remove action from possible moves set for this game
            possible_moves.remove(a)

            if r[1] == False:
                #b.invert()
                #make random move for O's
                ao = random.choice(list(possible_moves))
                r = b.make_move(ao)
                possible_moves.remove(ao)
            #save new state
            ss = b.get_vec()
            game_state = r
            replay.append({"state0": s,
                           "state1": ss,
                           "action": a,
                           "q": p,
                           "reward": r[0],
                           "terminal": r[1]})

        log.info("Game %s ended in %s steps. %s", game_number, step, game_result_map[game_state[0]])
        log.info("\n%s",b)
        print(replay)
        game_number = game_number + 1
        if game_state[0] == 1:
            wins = wins + 1
        log.info("Wining rate: %s", wins/game_number)
        y = simple_dqn.calc_target(dqn, replay, gamma)
        simple_dqn.train(dqn, replay, y)

        #decrease epsilon with every step
        if epsilon > epsilon_min:
            epsilon = epsilon - epsilon_step
    except KeyboardInterrupt:
        sys.exit(1)

