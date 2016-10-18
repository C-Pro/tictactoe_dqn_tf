import tensorflow
import tflearn as tf
import numpy as np
import os

def getModel(model_name):
    """Build a neural network to approximate Q values
    for every possible action given board state"""
    n = 9 # 9 is number of cells on tictactoe board

    g = tensorflow.Graph()
    with g.as_default():
        tnorm = tf.initializations.uniform(minval=-1.0, maxval=1.0)
        net = tf.input_data(shape=[None, 3, 3, 1])
        net = tf.conv_2d(net, n, 3, 1, activation='relu', regularizer='L2', weights_init=tnorm)
        #net = tf.conv_2d(net, n, 3, 1, activation='relu', regularizer='L2', weights_init=tnorm)
        #net = tf.conv_2d(net, n, 3, 1, activation='relu', regularizer='L2', weights_init=tnorm)
        #net = tf.fully_connected(net, n*n, activation='linear', regularizer='L2', weights_init=tnorm)
        #net = tf.fully_connected(net, 6*n, activation='linear', regularizer='L2', weights_init=tnorm)
        net = tf.fully_connected(net, 3*n, activation='linear', regularizer='L2', weights_init=tnorm)
        net = tf.fully_connected(net, n, activation='linear', regularizer='L2', weights_init=tnorm)
        net = tf.regression(net, optimizer='rmsprop', loss='mean_square', learning_rate=0.001)
        model = tf.DNN(net)

        if os.path.isfile(model_name):
            model.load(model_name)

    return (model, g)

def reshape(batch):
    """Reshape batch for conv2d input"""
    return np.reshape(batch, (-1, 3, 3, 1))


def train(model, g, minibatch, y, game_number, model_name, n_epoch=10):
    """"Train a dnn model given minibatch of transitions and
    corresponding precomputed Q values
    :param model: tf.DNN model representing Q-funcion
    :param minibatch: batch of transitions to learn on
    :param Y: target values for corresponding destination states in minibatch"""
    name = model_name + str(game_number).zfill(5)
    x = reshape([s["state0"] for s in minibatch])
    model.fit(x, y, n_epoch=n_epoch, validation_set=0.01, shuffle=True, run_id=name)
    with g.as_default():
        model.save(model_name)


def calc_target(model, minibatch, gamma):
    """Calculate target values (Y) using Bellman equation for minibatch
    :param model: tf.DNN model representing Q-funcion
    :param minibatch: batch of transitions to learn on"""
    y = []
    for ex in minibatch:
        q_values = model.predict(reshape(ex["state0"]))[0]
        #Mark all already occupied cells as "bad"
        for (i, cell) in enumerate(ex["state0"]):
            if cell != 0:
                q_values[i] = -1.0
        if ex["terminal"]:
            q_values[ex["action"]] = ex["reward"]
        else:
            q_values[ex["action"]] = ex["reward"] + gamma * np.max(model.predict(reshape(ex["state1"]))[0])
        y.append(q_values)
    return y

