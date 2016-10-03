import board
import simple_dqn
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
epsilon_min=0.01
epsilon_step=(epsilon-epsilon_min)/100
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
        inverted_rep.append({"s": invert_vector(replay["s"]),
                             "ss": invert_vector(replay["ss"]),
                             "action": replay["a"],
                             "reward": replay["reward"]
                             })
    return inverted_rep

def play_move(model, board, possible_moves):
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
        #make a random move or use dqn prediction
        if random.random() < epsilon:
            #a is an action(step) to be performed
            a = random.choice(list(possible_moves))
        else:
            #select best action based on model prediction
            a = np.argmax(p)
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


while True:
    replay = []

    b = board.Board()
    #game state is pair of Reward,Terminal
    #where reward is 1 when X win, -1 when O win, and 0 otherwise
    #Terminal is true when the game is ended
    game_state = (0, False)
    #game step counter
    step = 0

    try:
        log.info("Starting game %d",game_number)
        possible_moves = set([i for i in range(0,9)])
        #repeat until the game is ended
        while game_state[1] == False:
            step = step + 1
            s = b.get_vec()
            #play next move as X
            (a, r, ss) = play_move(dqn, b, possible_moves)           
            if r[1] == False:
                #play next move as O
                (aO, r, ss) = play_move(dqn, b, possible_moves)
            game_state = r
            replay.append({"state0": s,
                           "state1": ss,
                           "action": a,
                           "reward": r[0],
                           "terminal": r[1]})

        log.info("Game %s ended in %s steps. %s", game_number, step, game_result_map[game_state[0]])
        log.info("\n%s",b)
        game_number = game_number + 1
        if game_state[0] == 1:
            wins = wins + 1
        if game_state[0] == -1:
            losses = losses + 1
        log.info("Wining rate: %s", wins/game_number)
        log.info("Losing rate: %s", losses/game_number)

        random.shuffle(replay)
        # Use Bellman equation and model prediction to backpropagate reward
        y = simple_dqn.calc_target(dqn, replay, gamma)
        # Train model on replay        
        simple_dqn.train(dqn, replay, y, game_number)

        # Decrease epsilon with every step
        if epsilon > epsilon_min:
            epsilon = epsilon - epsilon_step
    except KeyboardInterrupt:
        sys.exit(1)

