import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mnist


def create_mlp(input_shape, n_outputs):
    network = keras.models.Sequential()

    network.add(keras.layers.InputLayer(input_shape))
    network.add(keras.layers.Flatten())
    network.add(keras.layers.Dense(256, activation='relu'))
    network.add(keras.layers.Dense(n_outputs, activation='softmax'))
    return network


(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = mnist.load_data(nval=1000)

mlp = create_mlp(input_shape=(28, 28, 1), n_outputs=10)
sgd = keras.optimizers.SGD(lr=1e-4)

mlp.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

logs = mlp.fit(train_images, train_labels, batch_size=16, epochs=1, verbose=1, validation_data=(val_images, val_labels))


# Exercise 1
fig, ax = plt.subplots(1, 2)
ax[0].plot(logs.history['loss'])
ax[0].plot(logs.history['val_loss'])
ax[0].legend(['loss', 'val_loss'])
ax[1].plot(logs.history['acc'])
ax[1].plot(logs.history['val_acc'])
ax[1].legend(['acc', 'val_acc'])
plt.show()


# Inspecting weights and layer outputs
layers = mlp.layers
[W1, b1] = layers[1].get_weights()
[W2, b2] = layers[2].get_weights()

layer1 = mlp.layers[1]
layer1_output = keras.backend.function([mlp.input], [layer1.output])
intermediate_output = layer1_output(train_images[39][None])[0]



