__author__ = 'Saeid SOHILY-KHAH'
"""
Autoencoder: Variational Autoencoder (VAE) based on a fully-connected neural network [using Keras] (MNIST) 
"""
import keras
import numpy as np
import keras.backend as kb
from scipy.stats import norm
import matplotlib.pyplot as plt


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


# Define negative log likelihood loss function
def neg_loglikelihood(y_true, y_pred):
    '''
    Define loss function
    :param y_true:
    :param y_pred:
    :return:
    '''
    bc = keras.losses.binary_crossentropy(y_true, y_pred) # binary_crossentropy gives the mean over the last axis
    return kb.sum(bc, axis=-1) # return sum of means


# Transform layer that adds KL-divergence to the model loss
class KL_divergence_layer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KL_divergence_layer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * kb.sum(1 + log_var - kb.square(mu) - kb.exp(log_var), axis=-1)
        self.add_loss(kb.mean(kl_batch), inputs=inputs)
        return inputs


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load mnist digits dataset
    X_train, y_train, X_test, y_test, img_size = load_data()

    # TensorFlow: Initialization
    autoencoder_units = 128
    in_out = img_size * img_size
    latent_dim = 2
    iter_number = 50  # number of iterations of training steps
    batch_size = 256  # to use only ... size data to train at each iteration of the optimizer

    # TensorFlow: Create layers
    # Define encoder network which turns input samples x into two parameters in a latent space
    input_fn = keras.layers.Input(shape=(in_out,))
    encoder_layer = keras.layers.Dense(autoencoder_units, activation='relu')(input_fn)

    z_mu = keras.layers.Dense(latent_dim)(encoder_layer)  # mean parameter in latent space
    z_log_sigma = keras.layers.Dense(latent_dim)(encoder_layer)  # log_sigma parameter in latent space
    z_mu, z_log_sigma = KL_divergence_layer()([z_mu, z_log_sigma])  # kl-divergence

    # Define sampling function to sample similar points z from latent normal distribution [z = z_mean + exp(z_log_sigma) * epsilon]
    z_sigma = keras.layers.Lambda(lambda t: kb.exp(.5 * t))(z_log_sigma)
    eps = keras.layers.Input(tensor=kb.random_normal(shape=(kb.shape(input_fn)[0], latent_dim)))
    z_eps = keras.layers.Multiply()([z_sigma, eps])
    z = keras.layers.Add()([z_mu, z_eps])

    # Define decoder network which maps the latent space points back to original input
    decoder_layer = keras.models.Sequential([
        keras.layers.Dense(autoencoder_units, input_dim=latent_dim, activation='relu'),
        keras.layers.Dense(in_out, activation='sigmoid')
    ])

    # TensorFlow: Model
    # Encoder model
    encoder = keras.models.Model(input_fn, z_mu)

    # Variational autoencoder (VAE) model
    vae = keras.models.Model(inputs=[input_fn, eps], outputs=decoder_layer(z))
    x_pred = decoder_layer(z)

    # TensorFlow: Compile
    vae.compile(optimizer='rmsprop', loss=neg_loglikelihood)

    # TensorFlow: Training mode
    vae.fit(X_train,
            X_train,
            shuffle=True,
            epochs=iter_number,
            batch_size=batch_size,
            validation_data=(X_test, X_test))

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the digit classes in the latent space
    z_test = encoder.predict(X_test, batch_size=batch_size)
    im = axes[0].scatter(z_test[:, 0], z_test[:, 1], c=y_test, alpha=.5, s=10, cmap='viridis')
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title('MNIST DIGITS IN LATENT SPACE', fontsize=12)

    # plot 2D manifold of the digits
    fig_dim = 20  # figure with 20x20 digits
    u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, fig_dim), np.linspace(0.05, 0.95, fig_dim)))
    z_grid = norm.ppf(u_grid)
    x_decoded = decoder_layer.predict(z_grid.reshape(fig_dim * fig_dim, 2))
    x_decoded = x_decoded.reshape(fig_dim, fig_dim, img_size, img_size)
    axes[1].imshow(np.block(list(map(list, x_decoded))), cmap='Blues')
    axes[1].set_title('MNIST DIGITS MANIFOLD', fontsize=12)

    # To save the plot locally
    plt.savefig('keras_variational_autoencoder.png', bbox_inches='tight')
    plt.show()
