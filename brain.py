from tensorflow import keras
from secrets import randbelow
import random
import numpy as np
import time

# - bellman function to calc loss and fit model. q(s,a) = r + y * max(q(s',a'))
# - state, action, reward, state'
# - state -> model -> q value per action

def create_brain(conf_b):
    result = brain(conf_b)
    if int(conf_b['autoload']) == 1:
        try:
            result.load()
        except:
            result.save()

    return result

class brain:
    def __init__(self, conf_b):
        self.cp_path = conf_b['save_filename']
        self.cp_callback = keras.callbacks.ModelCheckpoint(self.cp_path, verbose=0)

        self.discount_factor = float(conf_b['discount_factor'])
        self.rand_thresh = float(conf_b['random_action_thresh'])
        self.memory = []
        self.memory_max = int(conf_b['memory_buffer'])
        self.mini_batch_size = int(conf_b['minibatch_size'])
        self.learning_rate = float(conf_b['learning_rate'])

        self.input_size = int(conf_b['input_size'])
        self.n_frames = int(conf_b['frame_buffer'])
        self.verbose = int(conf_b['verbose'])

        self.round = 0

        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential()

        # convolutions
        # learn features in each layer
        model.add(keras.layers.Conv3D(16, (1, 5, 5), strides=(1, 2, 2), activation='relu', input_shape=(self.n_frames, self.input_size, self.input_size, 1)))
        # learn features between layers
        model.add(keras.layers.Conv3D(32, (2, 3, 3), strides=(2, 2, 2), activation='relu'))
        # pooling
        model.add(keras.layers.Conv3D(64, (1, 3, 3), strides=(1, 2, 2), activation='relu'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization(axis=-1))
        model.add(keras.layers.Dropout(0.2))

        # fully connected
        model.add(keras.layers.Dense(128, activation='relu'))
        #model.add(keras.layers.Dense(64, activation='relu'))
        # output
        model.add(keras.layers.Dense(4, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def think(self, state):
        if randbelow(10) < self.rand_thresh:
            return randbelow(4)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state_a, action, reward, state_b, ended):
        if len(self.memory) > self.memory_max:
            self.memory = []
        self.memory.append((state_a, action, reward, state_b, ended))

    def train(self, state_a, action, reward, state_b, ended):
        a_q = self.model.predict(state_a)
        b_q = self.model.predict(state_b)

        # inject new information about uncovered rewards
        if not ended:
            reward = reward + self.discount_factor * np.amax(b_q)

        # fit action value function to new q values
        a_q[0][action] = reward
        self.model.fit(state_a, a_q, epochs=1, verbose=self.verbose)

    def short_memory_training(self, state_a, action, reward, state_b, ended):
        self.train(state_a, action, reward, state_b, ended)

    def long_memory_training(self):
        # train on random minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.mini_batch_size))

        self.rand_thresh = self.rand_thresh * 0.985
        for sample in minibatch:
            # self.memory.append((state_a, action, reward, state_b, ended))
            self.train(sample[0], sample[1], sample[2], sample[3], sample[4])

    def save(self):
        self.model.save_weights(self.cp_path)
        #print("model saved")

    def load(self):
        self.model.load_weights(self.cp_path)
        print("model loaded")

    def summary(self):
        self.model.summary()
