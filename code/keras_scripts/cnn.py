import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mnist


def create_cnn(input_shape, n_outputs):
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape))
    model.add(keras.layers.Conv2D(16, (7, 7), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_outputs, activation='softmax'))
    return model


(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = mnist.load_data(nval=1000)

cnn = create_cnn(input_shape=(28, 28, 1), n_outputs=10)
sgd = keras.optimizers.SGD(lr=1e-2)

cnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

logs = cnn.fit(train_images, train_labels, batch_size=32, epochs=20, verbose=1, validation_data=(val_images, val_labels))


# Visualize kernels
[W0, b0] = cnn.layers[0].get_weights()
fig, ax = plt.subplots(4, 4)
for i in range(16):
    ax.flatten()[i].imshow(W0[:, :, 0, i])

[x.set_axis_off() for x in ax.flatten()]
plt.show()

