Snake Deep Q AI
=============
![alt text](https://raw.githubusercontent.com/elodea/Deep-Q-Snake/master/img/game.png)

Architecture
=============
Architecture similar to 'Playing Atari with Deep Reinforcement Learning' (2013) V Mnih, K Kavukcuoglu, D Silver et al.

The game's pixels are used directly as network input. 400 x 400 RGB game screen greyscaled and downsampled to 84 x 84. 4 frames stacked and fed into a 3d convolution

![alt text](https://raw.githubusercontent.com/elodea/Deep-Q-Snake/master/img/processed.png)

Three convolutional layers, one fully connected, and batch norm/dropout to help against overfitting
1. Conv [16 1x5x5 filters, stride 1x2x2, relu] - features in each layer
2. Conv [32 2x3x3 filters, stride 2x2x2, relu] - features between layers
3. Conv [64 1x3x3 filters, stride 1x2x2, relu] - conv pooling
4. Batch Normalisation
5. Dropout [0.2]
6. Dense [128, relu]
7. Dense [4, softmax]

Replay Training
==============
1. Long term memory training
- End of each game round, randomly select a mini batch of state transition memories (old_state, action, new_state, reward) to train the network
2. Short term memory training
- Train the last state transition

Config
===============
See config.ini

Dependencies
===============
Keras, CV2, Numpy, matplotlib, configparser
