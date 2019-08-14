from tensorflow import keras
from secrets import randbelow
import random
import numpy as np


# - bellman function to calc loss and fit model. q(s,a) = r + y * max(q(s',a'))
# - state, action, reward, state'
# - state -> model -> q value per action

class brain:
    def __init__(self, w, h):
        self.cp_path = 'saved_weights.h5'
        # cp_dir = os.path.dirname(cp_path)
        # print("cp_dir: ", cp_dir)
        self.cp_callback = keras.callbacks.ModelCheckpoint(self.cp_path, save_weights_only=True, verbose=0)

        self.reward_multi = 5
        self.discount_factor = 0.9
        self.rand_thresh = 5
        self.memory = []
        self.mini_batch_size = 250
        self.learning_rate = 0.0006

        self.model = self.create_model(w, h)

    def create_model(self, w_input, h_input):
        model = keras.Sequential()

        # convolutions
        model.add(keras.layers.Conv2D(12, (8, 8), strides=(4, 4), activation='relu', input_shape=(w_input, h_input, 1)))
        model.add(keras.layers.Conv2D(18, (4, 4), strides=(2, 2), activation='relu'))  # inbuilt pooling
        model.add(keras.layers.Flatten())
        # fully connected
        model.add(keras.layers.Dense(128, activation='relu'))
        #model.add(keras.layers.BatchNormalization(axis=-1))
        model.add(keras.layers.Dropout(0.2))
        #model.add(keras.layers.Dense(64, activation='relu'))
        # output
        model.add(keras.layers.Dense(4, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def think(self, state):
        if randbelow(10) > self.rand_thresh:
            return randbelow(4)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state_a, action, reward, state_b, ended):
        self.memory.append((state_a, action, reward, state_b, ended))

    def train(self, state_a, action, reward, state_b, ended, v_opt=0):
        #scale reward
        reward = reward * self.reward_multi
        # inject new information about uncovered rewards
        if not ended:
            reward = reward + self.discount_factor * np.amax(self.model.predict(state_b))
        else:
            reward = 0

        # fit action value function to new q values
        actual_q = self.model.predict(state_a)
        actual_q[0][action] = reward
        self.model.fit(state_a, actual_q, epochs=5, verbose=v_opt)

    def short_memory_training(self, state_a, action, reward, state_b, ended):
        if (reward > 0):
            print("reward found!")
            self.train(state_a, action, reward, state_b, ended)

    def long_memory_training(self):
        # train on random minibatch
        minibatch = random.sample(self.memory, min(len(self.memory), self.mini_batch_size))

        print("training minibatch: ", self.mini_batch_size)
        for sample in minibatch:
            # self.memory.append((state_a, action, reward, state_b, ended))
            self.train(sample[0], sample[1], sample[2], sample[3], sample[4], 0)

    def save(self):
        self.model.save_weights(self.cp_path)
        print("model saved")

    def load(self):
        self.model.load_weights(self.cp_path)
        print("model loaded")

    def summary(self):
        self.model.summary()


'''
def draw():
    plt.figure(figsize=(6,6))
    for i in range(25):
        plt.subplot(5, 5, i+1) #select subplot
        plt.imshow(train_img[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_label[i]])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

    plt.show()
'''
