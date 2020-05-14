Example codes and experiments around autoencoder algorithms using TensorFlow, Keras, ...

Autoencoders are data compression algorithms which convert multi dimensional data (high precision, slow performance) to low dimensional data (high performance). 
They are neural networks algorithms that apply backpropagation in setting target value to be equal to input with minimum possible error.
Autoencoders are data-specific, lossy algorithms and learned automatically from data examples.
To build an autoencoder, one needs encoding, decoding and a loss function. 
Note that the size of encoder and decoder layer(s) is smaller than input layer, and the input’s dimensionality should be equal to output’s dimensionality.

The applications of autoencoders are: dimensionality reduction (specially for visualization), data denoising (e.g. image, audio) as well as super-resolution, image generation, colouring and processing (compression), watermark removal, feature extraction, recommendation systems, information retrieval and semantic hashing, anomaly detection, machine translation, popularity prediction, population synthesis, drug discovery, etc.
There are, basically, 7 types of autoencoders: denoising, sparse, contractive, undercomplete, deep, convolutional and variational Autoencoder.


- Deep fully connected autoencoder:

- Deep convolutional autoencoder:

- Image denoising autoencoder:

- Variational autoencoder (VAE):
Variational autoencoder is a generative model, where instead of mapping input to a fixed vector in latent bottleneck, maps input on to a distribution.
More simply, instead of letting the neural network learns an arbitrary function, tries to learn the parameters of a probability distribution which models the data.
So, in variational autoencoders, the normal bottleneck is replaced by two vectors: mean and standard deviation. 
Additionally, variational autoencoders have an extra term in the loss function: KL divergence, where the aim is be sure that the distribution that we are learning, is not too far from normally distributed.
In essence, KL-divergence is a measure of the difference between two probability distributions.


