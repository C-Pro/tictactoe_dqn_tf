import tflearn as tf
import numpy as np
import os

def getModel(model_name="tictactoe.model"):
    """Build a neural network to approximate Q values
    for every possible action given board state"""

    n = 9 # 9 is number of cells on tictactoe board
    #tnorm = tf.initializations.uniform(minval=-1.0, maxval=1.0)
    net = tf.input_data(shape=[None, n])    
    net = tf.fully_connected(net, n, activation='sigmoid', regularizer='L2')
    net = tf.fully_connected(net, n, activation='sigmoid', regularizer='L2') 
    net = tf.fully_connected(net, n, activation='sigmoid', regularizer='L2')
    #net = tf.fully_connected(net, n, activation='relu', regularizer='L2')
    #net = tf.fully_connected(net, n, activation='relu', regularizer='L2')
    #net = tf.fully_connected(net, n, activation='relu', regularizer='L2')
    net = tf.regression(net)

    model = tf.DNN(net)

    if os.path.isfile(model_name):
        model.load(model_name)

    # return model
    return model


def train(model, minibatch, y, game_number, model_name="tictactoe.model"):
    """"Train a dnn model given minibatch of transitions and
    corresponding precomputed Q values
    :param model: tf.DNN model representing Q-funcion
    :param minibatch: batch of transitions to learn on
    :param Y: target values for corresponding destination states in minibatch"""
    name = "tictactoe" + str(game_number).zfill(5)
    x = [s["state0"] for s in minibatch]
    model.fit(x, y, n_epoch=50, run_id=name)
    model.save(model_name)


def calc_target(model, minibatch, gamma):
    """Calculate target values (Y) using Bellman equation for minibatch
    :param model: tf.DNN model representing Q-funcion
    :param minibatch: batch of transitions to learn on"""
    y = []
    for ex in minibatch:
        q_values = model.predict([ex["state0"]])[0]
        #Mark all already occupied cells as "bad"
        for (i, cell) in enumerate(ex["state0"]):
            if cell != 0:
                q_values[i] = -0.1
        if ex["terminal"]:
            q_values[ex["action"]] = ex["reward"]
        else:
            q_values[ex["action"]] = ex["reward"] + gamma * np.max(model.predict([ex["state1"]])[0])
        y.append(q_values)
    return y


def calc_target2(model, minibatch, gamma):
    """Calculate target values (Y) by manually asigning time-decreasing revard to all minibatch steps
    :param model: tf.DNN model representing Q-funcion
    :param minibatch: batch of transitions to learn on"""
    y = []
    minibatch.reverse()
    for ex in minibatch:
        q_values = model.predict([ex["state0"]])[0]
        #Mark all already occupied cells as "bad"
        for (i, cell) in enumerate(ex["state0"]):
            if cell != 0:
                q_values[i] = -0.1
        if ex["terminal"]:
            q = ex["reward"]
        else:
            q = ex["reward"] + gamma * q
        q_values[ex["action"]] = q
        y.append(q_values)
    return y
