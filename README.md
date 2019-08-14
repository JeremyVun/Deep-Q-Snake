Snake Deep Q AI
=============
![alt text](https://raw.githubusercontent.com/elodea/Deep-Q-Snake/master/img/game.png)

Architecture
=============
Architecture similar to 'Playing Atari with Deep Reinforcement Learning' (2013) V Mnih, K Kavukcuoglu, D Silver et al.

The game's pixels are used directly as network input. 400 x 400 RGB game screen greyscaled and downsampled to 84 x 84

![alt text](https://raw.githubusercontent.com/elodea/Deep-Q-Snake/master/img/processed.png)

Two convolutional layers, one fully connected, and one dropout to help against overfitting
1. Conv [12 8x8 filters, stride 4, relu]
2. Conv [18 4x4 filters, stride 2, relu]
3. Dense [128, relu]
4. Dropout [0.2]
5. Dense [4, softmax]

Replay Training
==============
Two training triggers. Aim to train only events that are significant

1. End of each game round, randomly select a mini batch of state transition memories (old_state, action, new_state, reward) to train the network
2. Train the state transition whenever the agent dies or recieves a reward
