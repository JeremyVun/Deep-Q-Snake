import tensorflow as tf
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
            print("model loaded")
        except:
            result.save()
            print("new model created")

    return result

class brain:
    def __init__(self, conf_b):
        self.cp_path = conf_b['save_filename']
        self.cp_callback = keras.callbacks.ModelCheckpoint(self.cp_path, verbose=0)

        self.discount_factor = float(conf_b['discount_factor'])
        self.rand_action_thresh = float(conf_b['init_random_action_thresh'])
        self.min_random_action_thresh = float(conf_b['min_random_action_thresh'])
        self.rand_action_decay = float(conf_b['random_action_decay'])
        
        self.memory = []
        self.memory_max = int(conf_b['memory_buffer'])
        self.mini_batch_size = int(conf_b['minibatch_size'])
        self.learning_rate = float(conf_b['learning_rate'])
        self.epochs = int(conf_b['epochs'])

        self.input_size = int(conf_b['input_size'])
        self.n_frames = int(conf_b['frame_buffer'])
        self.verbose = int(conf_b['verbose'])
        self.autosave = bool(conf_b['autosave'])

        self.round = 0

        self.model = self.create_model()

    def get_batch_size(self):
        return min(len(self.memory), self.mini_batch_size)

    def get_action_thresh(self):
        return self.rand_action_thresh

    def get_epochs(self):
        return self.epochs

    def create_model(self):
        model = keras.Sequential()

        # convolutions
        # learn features in each layer
        model.add(keras.layers.Conv3D(12, (1, 8, 8), strides=(1, 4, 4), activation='relu', input_shape=(self.n_frames, self.input_size, self.input_size, 1)))
        model.add(keras.layers.Conv3D(24, (1, 4, 4), strides=(1, 2, 2), activation='relu'))

        model.add(keras.layers.Flatten())
        # model.add(keras.layers.BatchNormalization(axis=-1))
        model.add(keras.layers.Dropout(0.1))

        # fully connected
        model.add(keras.layers.Dense(192, activation='relu'))
        # model.add(keras.layers.Dense(128, activation='relu'))
        # output
        model.add(keras.layers.Dense(4, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def think(self, state):
        if randbelow(10) < self.rand_action_thresh:
            return randbelow(4)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state_a, action, reward, state_b, ended):
        if len(self.memory) >= self.memory_max:
            self.memory[randrange(self.memory_max)] = (state_a, action, reward, state_b, ended)
        else:
            self.memory.append((state_a, action, reward, state_b, ended))

    def train(self, state_a, action, reward, state_b, ended):
        a_q = self.model.predict(state_a)
        b_q = self.model.predict(state_b)

        # inject new information about uncovered rewards
        if not ended:
            reward = reward + self.discount_factor * np.amax(b_q)

        # fit action value function to new q values
        a_q[0][action] = reward
        self.model.fit(state_a, a_q, epochs=self.epochs, verbose=self.verbose)

    def short_memory_training(self, state_a, action, reward, state_b, ended):
        self.train(state_a, action, reward, state_b, ended)

    def long_memory_training(self):
        #decay random action threshhold
        self.rand_action_thresh = max(self.min_random_action_thresh, self.rand_action_thresh * self.rand_action_decay)

        if len(self.memory) > self.mini_batch_size:
            minibatch = self.memory[:self.mini_batch_size]
            self.memory = self.memory[self.mini_batch_size:]
            for sample in minibatch:
                self.train(sample[0], sample[1], sample[2], sample[3], sample[4])

        else:
            for sample in self.memory:
                self.train(sample[0], sample[1], sample[2], sample[3], sample[4])

        if self.autosave:
            self.save()

    def save(self):
        self.model.save_weights(self.cp_path)

    def load(self):
        self.model.load_weights(self.cp_path)

    def summary(self):
        self.model.summary()
