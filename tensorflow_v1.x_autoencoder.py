__author__ = 'Saeid SOHILY-KHAH'
"""
Autoencoder: Tensorflow (MNIST) Autoencoder Implementation [using TensorFlow 1.x] 
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Load data
def load_data():
    '''
    Load MNIST dataset
    :return: mnist, img_size, num_classes:
    '''
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshaping
    img_size = X_train.shape[-1]
    X_train = X_train.reshape((-1, img_size * img_size))
    X_test = X_test.reshape((-1, img_size * img_size))

    # Normalization
    epsilon = 1e-6 # solve divison by zero
    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) + epsilon)  # normalization
    X_test = (X_test - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) + epsilon)  # normalization

    return X_train, X_test, y_train, y_test, img_size


# Generate a batch of data for training model
def next_batch(batch_size, data, labels):
    '''
    Generate a batch by returnning batch_size of random data samples and labels
    :param batch_size:
    :param data:
    :param labels:
    :return:
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = data[idx]
    batch_labels = labels[idx]
    return batch_data, batch_labels


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load mnist data
    X_train, X_test, y_train, y_test, img_size = load_data()

    # Autoencoder: Initialization
    n_inputs = img_size * img_size # flatten image
    n_hidden_layer1 = 128 # number of features in hidden layer1
    n_hidden_layer2 = 64  # number of features in hidden layer2
    n_hidden_layer3 = 32  # number of features in hidden layer3
    learning_rate = 0.01
    batch_size = 100
    n_epochs = 10000
    model_path = os.getcwd() + '/saved_model'  # to save trained model in the current directory

    # Autoencoder: Graph input
    X = tf.placeholder("float", [None, n_inputs])

    # Autoencoder: Graph weights and biases
    weights = {
        # Weights in encoder layers
        'encoder_h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_layer1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_layer1, n_hidden_layer2])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer3])),

        # Weigths in decoder layers
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_layer3, n_hidden_layer2])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer1])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden_layer1, n_inputs])),
    }

    biases = {
        # Biases in encoder layers
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_layer1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_layer2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_layer3])),

        # Biases in decoder layers
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_layer2])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_layer1])),
        'decoder_b3': tf.Variable(tf.random_normal([n_inputs])),
    }

    # Autoencoder: Define model
    # Autoencoder: Define encoder layers
    encoder_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
    encoder_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer1, weights['encoder_h2']), biases['encoder_b2']))
    encoder_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer2, weights['encoder_h3']), biases['encoder_b3']))

    # Autoencoder: Define decoder layers
    decoder_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer3, weights['decoder_h1']), biases['decoder_b1']))
    decoder_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer1, weights['decoder_h2']), biases['decoder_b2']))
    decoder_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer2, weights['decoder_h3']), biases['decoder_b3']))

    # Autoencoder: Reconstruction of inputs
    y_pred = decoder_layer3 # regenerated images
    y_true = X  # targets (or labels) are the input data in autoencoders

    # Autoencoder: Define cost function and optimizer
    cost_function = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) # square error
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_function)

    # TensorFlow: Create a session to run the defined tensorflow graph
    sess = tf.Session()  # create a session

    # TensorFlow: Initialize the variables
    init = tf.global_variables_initializer()

    # TensorFlow: Create and instance of train.Saver
    saver = tf.train.Saver()

    # TensorFlow: Create an object class for writting summaries
    file_writer = tf.summary.FileWriter(model_path, sess.graph)

    sess.run(init) # execute the initializer

    # Autoencoder: Training...
    loss_history = []
    for epoch in range(n_epochs):
        X_train_batch, y_train_batch = next_batch(batch_size, X_train, y_train)
        _, loss = sess.run([optimizer, cost_function], feed_dict={X: X_train_batch})
        loss_history.append(loss)

        # Summarize results
        if epoch % 100 == 0:
            print('Epoch:{:<5}   ->   Loss={:.3f}'.format(epoch, round(loss, 3)))

    # TensorFlow: Save the model
    saver.save(sess, model_path)

    # TensorFlow: Restore the saved model (uncomment the line below to restore the saved model)
    # saver.restore(sess, model_path')

    # Autoencoder: Evaluation
    prediction = sess.run(y_pred, feed_dict={X: X_test[:10]}) # 10 test samples to display

    # TensorFlow: Close the session
    sess.close()

    # Plot settings
    fig = plt.figure(figsize=(17, 7))  # set figure size
    axes = gridspec.GridSpec(nrows=2, ncols=1, wspace=0.15, hspace=0.15)  # set figure shape
    axes.update(top=0.95, left=0.05, right=0.95, bottom=0.05)  # set tight_layout()

    # plot the regenerated data vs original images
    inner_subplot = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=10, subplot_spec=axes[0], wspace=0.25, hspace=0.25)
    for col in range(10):
        ax = plt.Subplot(fig, inner_subplot[0, col]) # first row in subplot
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('ACTUAL '+str(y_test[col]), fontsize=12)  # title: image label
        ax.imshow(X_test[col].reshape(img_size, img_size), aspect='auto')  # show original digit's image
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_subplot[1, col]) # second row in subplot
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('GENERATED ' + str(y_test[col]), fontsize=12)  # title: image label
        ax.imshow(prediction[col].reshape(img_size, img_size), aspect='auto')  # show generated digit's image
        fig.add_subplot(ax)

    # plot the cost for the model
    inner_subplot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=axes[1])
    ax = plt.Subplot(fig, inner_subplot[0, 0])
    ax.plot(loss_history, label='{} = {:.3f}'.format('COST (TRAIN)', round(loss_history[-1], 3)), color='r')
    ax.set_xlim([0.0, n_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.set_title('COST FUNCTION (TRAINING)', fontsize=12)
    ax.legend(loc="upper right")
    fig.add_subplot(ax)

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_autoencoder.png', bbox_inches='tight')
    plt.show()