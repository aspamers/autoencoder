"""
Auto-encoder module.
"""

from keras.layers import Input
from keras.models import Model


class AutoEncoder:
    """
    A simple and lightweight auto-encoder implementation.
    """
    def __init__(self, encoder_model, decoder_model):
        # Set essential parameters
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        # Get input shape from encoder model
        self.input_shape = self.encoder_model.input_shape[1:]

        # Initialize auto-encoder model
        self.autoencoder = None
        self.__initialize_autoencoder_model()

    def compile(self, *args, **kwargs):
        """
        Configures the model for training.

        Passes all arguments to the underlying Keras model compile function.
        """
        self.autoencoder.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Train the auto-encoder.

        Passes all arguments to the underlying Keras model fit function.
        """
        self.autoencoder.fit(*args, **kwargs)

    def load_weights(self, checkpoint_path):
        """
        Load auto-encoder weights. This also affects the reference to the base
        and head models.

        :param checkpoint_path: Path to the checkpoint file.
        """
        self.autoencoder.load_weights(checkpoint_path)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the auto-encoder.

        Passes all arguments to the underlying Keras model evaluate function.
        """
        return self.autoencoder.evaluate(*args, **kwargs)

    def __initialize_autoencoder_model(self):
        """
        Create the auto-encoder structure using the supplied base and
        head model.
        """
        input_1 = Input(shape=self.input_shape)

        encoded = self.encoder_model(input_1)
        decoded = self.decoder_model(encoded)
        self.autoencoder = Model(input_1, decoded)
