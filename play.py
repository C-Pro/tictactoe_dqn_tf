import board
import simple_dqn
import numpy as np
import random
import logging
import sys

#game number counter
game_number=0
wins=0

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
            print(p)
            #select best action based on model prediction
            a = np.argmax(p)
            r = None
            while r == None:
                if step == 1:
                    a = 4
                #make chosen move
                r = b.make_move(a)
                #if move not possible - choose another
                if r == None:
                    a = random.choice(list(possible_moves))
                    continue
            if r[1]:
                game_state=r
                break
            #remove action from possible moves set for this game
            possible_moves.remove(a)
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

