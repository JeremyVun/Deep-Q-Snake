import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import os

class brain:
    def __init__(self):
        self.model = self.create_model()
        self.memory = []

    def create_model(self):
        model = keras.Sequential()

        #problem with this line
        model.add(keras.layers.Dense(512, activation='relu', input_shape=(784,)))

        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def think(self, state):
        print("brain thinking")
        return 0

    def remember(self, state_a, action, reward, state_b, ended):
        print("brain remember")
        self.memory.append((state_a, action, reward, state_b, ended))

    def short_memory_training(self, state_a, action, reward, state_b, ended):
        print("brain short memory training")

    def long_memory_training(self):
        print("brain long replay training")

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

fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_label), (test_img, test_label) = fashion_mnist.load_data()

#labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#flatten
train_img = train_img.reshape(-1,28*28) / 255.0
test_img = test_img.reshape(-1,28*28) / 255.0

#checkpoint
cp_path = "hello.ckpt"
cp_dir = os.path.dirname(cp_path)
print("cp_dir: ", cp_dir)

cp_callback = keras.callbacks.ModelCheckpoint(cp_path, save_weights_only=True, verbose=1)

model = create_model()
#model.summary()

#model.fit(train_img, train_label, callbacks=[cp_callback], period=1 epochs=5)
loss_and_metrics = model.evaluate(test_img, test_label)

model.load_weights(cp_path)
loss_and_metrics = model.evaluate(test_img, test_label)
'''