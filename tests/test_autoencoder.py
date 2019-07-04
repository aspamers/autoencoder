
"""
Tests for the auto-encoder module
"""

import numpy as np
import keras
from keras import Model, Input
from keras.layers import Dense, BatchNormalization, Activation

from autoencoder import AutoEncoder


def test_autoencoder():
    """
    Test that all components of the auto-encoder work correctly by executing a
    training run against generated data.
    """

    input_shape = (3,)
    epochs = 1000

    # Generate some data
    x_train = np.random.rand(100, 3)
    x_test = np.random.rand(30, 3)

    # Define encoder and decoder model
    def create_encoder_model(input_shape):
        model_input = Input(shape=input_shape)

        encoder = Dense(4)(model_input)
        encoder = BatchNormalization()(encoder)
        encoder = Activation(activation='relu')(encoder)

        return Model(model_input, encoder)

    def create_decoder_model(embedding_shape):
        embedding_a = Input(shape=embedding_shape)

        decoder = Dense(3)(embedding_a)
        decoder = BatchNormalization()(decoder)
        decoder = Activation(activation='relu')(decoder)

        return Model(embedding_a, decoder)

    # Create auto-encoder network
    encoder_model = create_encoder_model(input_shape)
    decoder_model = create_decoder_model(encoder_model.output_shape)
    autoencoder = AutoEncoder(encoder_model, decoder_model)

    # Prepare auto-encoder for training
    autoencoder.compile(loss='binary_crossentropy',
                        optimizer='adam')

    # Evaluate network before training to establish a baseline
    score_before = autoencoder.evaluate(x_train, x_train)

    # Train network
    autoencoder.fit(x_train, x_train,
                    validation_data=(x_test, x_test),
                    epochs=epochs)

    # Evaluate network
    score_after = autoencoder.evaluate(x_train, x_train)

    # Ensure that the training loss score improved as a result of the training
    assert(score_before > score_after)
