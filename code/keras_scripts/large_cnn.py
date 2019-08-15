import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mnist


def create_larger_cnn(input_shape, n_outputs):
    network = keras.models.Sequential()

    network.add(keras.layers.InputLayer(input_shape))
    network.add(keras.layers.Conv2D(32, (3, 3)))
    network.add(keras.layers.Conv2D(32, (3, 3)))
    network.add(keras.layers.MaxPooling2D((2, 2)))

    network.add(keras.layers.Conv2D(64, (3, 3)))
    network.add(keras.layers.Conv2D(64, (3, 3)))
    network.add(keras.layers.MaxPooling2D((2, 2)))

    network.add(keras.layers.Flatten())
    network.add(keras.layers.Dense(n_outputs, activation='softmax'))
    return network


(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = mnist.load_data(nval=1000)

cnn = create_larger_cnn(input_shape=(28, 28, 1), n_outputs=10)
sgd = keras.optimizers.SGD(lr=1e-4)

cnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

logs = cnn.fit(train_images, train_labels, batch_size=16, epochs=20, verbose=1, validation_data=(val_images, val_labels))

