from tensorflow import keras


def load_data(nval):
    """
    Load the MNIST data set.

    Args:
        nval (int): The size of the validation set.
    Returns:
        (train_images, train_labels), (val_images, val_labels), (test_images, test_labels): Numpy arrays containing images and labels.
    """
    # Load the MNIST data train and test sets
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Convert the train labels to one-hot encoding, for example if the
    # label is 3, the one-hot encoding is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], if
    # the label is 5, the one-hot  encoding is [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    # Devide the train set into a train and val set,
    val_images = train_images[:nval].copy()
    train_images = train_images[nval:].copy()
    val_labels = train_labels[:nval].copy()
    train_labels = train_labels[nval:].copy()

    # Add an extra axis to the arrays of images, i.e. the shape is initially
    # (10000, 28, 28), it now becomes (10000, 28, 28, 1). This is required,
    # because Keras wants to know how many channels there are in the images.
    # Because The images are grayscale, we only have one. For RGB color image,
    # the shape would end in 3.
    val_images = val_images[:, :, :, None]
    train_images = train_images[:, :, :, None]
    test_images = test_images[:, :, :, None]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data(nval=1000)
    for im, lab in zip(train_images, train_labels):
        plt.imshow(im[:, :, 0])
        plt.title('Labels: {}\nOne-hot encoding: {}'.format(np.argmax(lab), lab))
        plt.show()
