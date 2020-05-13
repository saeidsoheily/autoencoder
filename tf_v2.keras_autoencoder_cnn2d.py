__author__ = 'Saeid SOHILY-KHAH'
"""
Autoencoder: Deep autoencoder based on a 2d-convolutional neural network [using TensorFlow 2.x] (MNIST) 
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

    # 2D reshapping data
    X_train = X_train.reshape((len(X_train), img_size, img_size, 1))
    X_test = X_test.reshape((len(X_test), img_size, img_size, 1))

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
    en_conv_layer_units = [16, 8] # units in encoder convolutional layers
    de_conv_layer_units = [8, 16] # units in decoder convolutional layers
    conv_kernel_size = [3, 3] # 2D convolution window along (height,width) [tuple/list of 2 int or 1 int for both dims]
    strides_size = 1 # strides of the convolution along the height and width
    maxp_pool_size = upsa_size = [2, 2]
    in_out_size = (img_size , img_size, 1)
    iter_number = 10  # number of iterations of training steps
    batch_size = 128  # to use only ... size data to train at each iteration of the optimizer

    # TensorFlow: Create layers
    input_fn = tf.keras.layers.Input(shape=(in_out_size))  # placeholder tf v1.x changed to layers.Input tf v2.x
    encoder_layer1_conv = tf.keras.layers.Conv2D(filters=en_conv_layer_units[0],
                                                 kernel_size=conv_kernel_size, # 2d convolution window
                                                 strides=strides_size,
                                                 activation='relu',
                                                 padding='same')(input_fn) # encode input representation

    encoder_layer1_maxp = tf.keras.layers.MaxPooling2D(pool_size=maxp_pool_size,
                                                       padding='same')(encoder_layer1_conv)

    encoder_layer2_conv = tf.keras.layers.Conv2D(filters=en_conv_layer_units[1],
                                                 kernel_size=conv_kernel_size, # 2d convolution window
                                                 strides=strides_size,
                                                 activation='relu',
                                                 padding='same')(encoder_layer1_maxp)

    encoder_layer2_maxp = tf.keras.layers.MaxPooling2D(pool_size=maxp_pool_size,
                                                       padding='same')(encoder_layer2_conv)

    decoder_layer1_conv = tf.keras.layers.Conv2D(filters=de_conv_layer_units[0],
                                                 kernel_size=conv_kernel_size, # 2d convolution window
                                                 strides=strides_size,
                                                 activation='relu',
                                                 padding='same')(encoder_layer2_maxp)

    decoder_layer1_upsa = tf.keras.layers.UpSampling2D(size=upsa_size)(decoder_layer1_conv)

    decoder_layer2_conv = tf.keras.layers.Conv2D(filters=de_conv_layer_units[1],
                                                 kernel_size=conv_kernel_size, # 2d convolution window
                                                 strides=strides_size,
                                                 activation='relu',
                                                 padding='same')(decoder_layer1_upsa)

    decoder_layer2_upsa = tf.keras.layers.UpSampling2D(size=upsa_size)(decoder_layer2_conv)

    decoder_layer3_conv = tf.keras.layers.Conv2D(filters=1,
                                                 kernel_size=conv_kernel_size, # 2d convolution window
                                                 strides=strides_size,
                                                 activation='sigmoid',
                                                 padding='same')(decoder_layer2_upsa)

    # TensorFlow: Model
    autoencoder_model = tf.keras.models.Model(input_fn, decoder_layer3_conv)  # maps an input to its reconstruction

    # TensorFlow: Compile
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    # TensorFlow: Training model
    autoencoder_model.fit(X_train, X_train, # self-supervized learning (no label)
                          epochs=iter_number,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test))

    # TensorFlow: Evaluation
    X_pred = autoencoder_model.predict(X_test)

    # Plot settings
    fig = plt.figure(figsize=(17, 5))  # set figure size
    axes = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.15, hspace=0.15)  # set figure shape
    axes.update(top=0.95, left=0.05, right=0.95, bottom=0.05)  # set tight_layout()

    # Plot sample actual data
    plot_sample_data(X_test, y_test, img_size, axes[0],
                     tag='ACTUAL ',
                     color_map='viridis',
                     onehotencoder_label=False)  # plot original image

    # Plot sample autoencoded data using autoencoder model
    plot_sample_data(X_pred, y_test, img_size, axes[1],
                     tag='_CODED ',
                     color_map='viridis',
                     onehotencoder_label=False)  # plot reconstructed image

    # To save the plot locally
    plt.savefig('tensorflow_keras_autoencoder_cnn2d.png', bbox_inches='tight')
    plt.show()