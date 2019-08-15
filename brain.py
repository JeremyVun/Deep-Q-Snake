from tensorflow import keras
from secrets import randbelow
import random
import numpy as np


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

        self.round = 0

        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential()

        # convolutions
        model.add(keras.layers.Conv2D(8, (5, 5), strides=(2, 2), activation='relu', input_shape=(self.input_size, self.input_size, self.n_frames)))
        model.add(keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu'))  # inbuilt pooling
        model.add(keras.layers.Flatten())
        # fully connected
        model.add(keras.layers.Dense(128, activation='relu'))
        #model.add(keras.layers.BatchNormalization(axis=-1))
        #model.add(keras.layers.Dropout(0.2))
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

    def train(self, state_a, action, reward, state_b, ended, v_opt=0):
        # inject new information about uncovered rewards
        if not ended:
            reward = reward + self.discount_factor * np.amax(self.model.predict(state_b))

        # fit action value function to new q values
        actual_q = self.model.predict(state_a)
        actual_q[0][action] = reward
        self.model.fit(state_a, actual_q, epochs=1, verbose=v_opt)

    def short_memory_training(self, state_a, action, reward, state_b, ended):
        self.train(state_a, action, reward, state_b, ended)

    def long_memory_training(self):
        # train on random minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.mini_batch_size))

        self.rand_thresh = self.rand_thresh * 0.985
        for sample in minibatch:
            # self.memory.append((state_a, action, reward, state_b, ended))
            self.train(sample[0], sample[1], sample[2], sample[3], sample[4], 0)

    def save(self):
        self.model.save_weights(self.cp_path)
        #print("model saved")

    def load(self):
        self.model.load_weights(self.cp_path)
        print("model loaded")

    def summary(self):
        self.model.summary()
