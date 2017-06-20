# DQN Tic-Tac-Toe #

My experiment on reinforcement learning (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) to train neural network to play 3x3 tic-tac-toe game.
It differs from classical DQN task because there is no 'game' or 'emulator' that plays against trained network, but another network that plays for the other side.
I use replay buffer and train two networks simultaneously (one for X's and one for O's). Before starting a round of games, two networks are randomly sampled from pool to promote exploration (epsilon with decay is also used).

Tested with python 3.5.2. Run `pip install -r requirements.txt` to install dependencies. If you do not have Nvidia GPU, remove `-gpu` suffix for tensorflow package in `requirements.txt` before installing with pip.

Use `learn.py` to train networks, `championship.py` to select best ones, `play.py` to play (need to input index of cell 0-8 and hit enter to make a move).
