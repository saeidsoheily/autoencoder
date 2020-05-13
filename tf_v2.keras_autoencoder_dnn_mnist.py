__author__ = 'Saeid SOHILY-KHAH'
"""
Autoencoder: Deep autoencoder based on a fully-connected neural network [using TensorFlow 2.x] (MNIST) 
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Load data
def load_data():
    '''
    Load MNIST dataset
    :return: X_train, y_train, X_test, y_test, img_size:
    '''
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_size = X_train.shape[-1]  # dimension of mnist image (i.e. 28)

    # Normalize data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Flatten data
    X_train = X_train.reshape((len(X_train), img_size * img_size))
    X_test = X_test.reshape((len(X_test), img_size * img_size))

    return X_train, y_train, X_test, y_test, img_size


# Plot sample data
def plot_sample_data(images, labels, img_size, axes, tag, color_map, onehotencoder_label=False):
    '''
    Plot 15 sample data from mnist dataset using gridspec
    :param images:
    :param labels:
    :param img_size:
    :param axes:
    :param tag: plot title
    :param color_map:
    :param onehotencoder_label:
    :return:
    '''
    # Plot original images
    inner_subplot = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=5, subplot_spec=axes, wspace=0.25, hspace=0.25)
    for row in range(3):
        for col in range(5):
            k = row * 3 + col
            ax = plt.Subplot(fig, inner_subplot[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            if onehotencoder_label:
                ax.set_title(tag + str(np.argmax(labels[k])), fontsize=10)  # title: image label
            else:
                ax.set_title(tag + str(labels[k]), fontsize=10)  # title: image label
            ax.imshow(images[k].reshape(img_size, img_size), aspect='auto', cmap=color_map)  # show digit's image
            fig.add_subplot(ax)
    return


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load mnist digits dataset
    X_train, y_train, X_test, y_test, img_size = load_data()

    # TensorFlow: Initialization
    autoencoder_units = 256
    en_layer_units = [128, 64] # units in encoder layers
    de_layer_units = [64, 128] # units in decoder layers
    in_out_size = img_size * img_size
    iter_number = 20  # number of iterations of training steps
    batch_size = 128  # to use only ... size data to train at each iteration of the optimizer

    # TensorFlow: Create layers
    input_fn = tf.keras.layers.Input(shape=(in_out_size,))  # placeholder tf v1.x changed to layers.Input tf v2.x

    encoder_layer1 = tf.keras.layers.Dense(autoencoder_units, activation='relu')(input_fn) # encode input representation
    encoder_layer2 = tf.keras.layers.Dense(en_layer_units[0], activation='relu')(encoder_layer1)
    encoder_layer3 = tf.keras.layers.Dense(en_layer_units[1], activation='relu')(encoder_layer2)

    decoder_layer1 = tf.keras.layers.Dense(de_layer_units[0], activation='relu')(encoder_layer3)
    decoder_layer2 = tf.keras.layers.Dense(de_layer_units[1], activation='relu')(decoder_layer1)
    decoder_layer3 = tf.keras.layers.Dense(units=in_out_size, activation='sigmoid')(decoder_layer2) # reconstruction

    # TensorFlow: Model
    encoder_model = tf.keras.models.Model(input_fn, encoder_layer3) # maps an input to its encoded representation
    autoencoder_model = tf.keras.models.Model(input_fn, decoder_layer3)  # maps an input to its reconstruction

    # TensorFlow: Compile
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    # TensorFlow: Training model
    autoencoder_model.fit(X_train, X_train, # self-supervized learning (no label)
                          epochs=iter_number,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test))

    # TensorFlow: Evaluation
    X_test_encoded = encoder_model.predict(X_test)
    X_pred = autoencoder_model.predict(X_test)

    # Plot settings
    fig = plt.figure(figsize=(21, 4))  # set figure size
    axes = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.15, hspace=0.15)  # set figure shape
    axes.update(top=0.95, left=0.05, right=0.95, bottom=0.05)  # set tight_layout()

    # Plot sample actual data
    plot_sample_data(X_test, y_test, img_size, axes[0],
                     tag='ACTUAL ',
                     color_map='viridis',
                     onehotencoder_label=False)  # plot original image

    # Plot sample encoded data
    plot_sample_data(X_test_encoded, y_test, int(np.sqrt(X_test_encoded.shape[1])), axes[1],
                     tag='ENCODED ',
                     color_map='viridis',
                     onehotencoder_label=False)  # plot encoded image

    # Plot sample autoencoded data using autoencoder model
    plot_sample_data(X_pred, y_test, img_size, axes[2],
                     tag='DECODED ',
                     color_map='viridis',
                     onehotencoder_label=False)  # plot reconstructed image

    # To save the plot locally
    plt.savefig('tensorflow_keras_autoencoder.png', bbox_inches='tight')
    plt.show()