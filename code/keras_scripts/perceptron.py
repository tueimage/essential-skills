import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import mnist


def create_perceptron(input_shape, n_outputs):
    network = keras.models.Sequential()

    network.add(keras.layers.InputLayer(input_shape))
    network.add(keras.layers.Flatten())
    network.add(keras.layers.Dense(n_outputs, activation='softmax'))
    return network


(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = mnist.load_data(nval=1000)

perceptron = create_perceptron(input_shape=(28, 28, 1), n_outputs=10)
sgd = keras.optimizers.SGD(lr=1e-4)

perceptron.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

logs = perceptron.fit(train_images, train_labels, batch_size=16, epochs=20, verbose=1, validation_data=(val_images, val_labels))


# Exercise 1
fig, ax = plt.subplots(1, 2)
ax[0].plot(logs.history['loss'])
ax[0].plot(logs.history['val_loss'])
ax[0].legend(['loss', 'val_loss'])
ax[1].plot(logs.history['acc'])
ax[1].plot(logs.history['val_acc'])
ax[1].legend(['acc', 'val_acc'])
plt.show()


# Exercise 2
predicted_values = [np.argmax(x) for x in perceptron.predict(test_images)]
true_values = [np.argmax(x) for x in test_labels]

confusion_matrix = np.zeros((10, 10))
for pred, true in zip(predicted_values, true_values):
    confusion_matrix[pred, true] += 1

plt.imshow(confusion_matrix)
plt.xlabel('True digit')
plt.ylabel('Network output')
plt.show()

