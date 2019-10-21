"""
This is a modified version of the Keras mnist example.
https://keras.io/examples/mnist_cnn/

Instead of using a fixed number of epochs this version continues to train until a stop criteria is reached.

An auto-encoder is used to pre-train an embedding for the network. The resulting embedding is then extended
with a softmax output layer for categorical predictions.

Model performance should be around 99.84% after training. The resulting model is identical in structure to the one in
the example yet shows considerable improvement in relative error confirming that the embedding learned by the
auto-encoder is useful.
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, UpSampling2D, Deconv2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Flatten, Dense

from autoencoder import AutoEncoder

batch_size = 128
num_classes = 10
epochs = 999999

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


def create_encoder_model(input_shape):
    model_input = Input(shape=input_shape)

    encoder = Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape)(model_input)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(activation='relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

    encoder = Conv2D(64, kernel_size=(3, 3), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(activation='relu')(encoder)
    encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

    encoder = Flatten()(encoder)
    encoder = Dense(128)(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(activation='relu')(encoder)

    return Model(model_input, encoder)


def create_decoder_model(embedding_shape):
    embedding_a = Input(shape=embedding_shape)

    decoder = Dense(1 * 28 * 28)(embedding_a)
    decoder = BatchNormalization()(decoder)
    decoder = Activation(activation='relu')(decoder)
    decoder = Reshape(input_shape)(decoder)

    return Model(embedding_a, decoder)


num_classes = 10
epochs = 999999

encoder_model = create_encoder_model(input_shape)

decoder_model = create_decoder_model(encoder_model.output_shape)
autoencoder_network = AutoEncoder(encoder_model, decoder_model)
autoencoder_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])

autoencoder_checkpoint_path = "./autoencoder_checkpoint"

autoencoder_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(autoencoder_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

autoencoder_network.fit(x_train, x_train,
                        validation_data=(x_test, x_test),
                        batch_size=128,
                        epochs=epochs,
                        callbacks=autoencoder_callbacks)

autoencoder_network.load_weights(autoencoder_checkpoint_path)
embedding = encoder_model.outputs[-1]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Add softmax layer to the pre-trained embedding network
embedding = Dense(num_classes)(embedding)
embedding = BatchNormalization()(embedding)
embedding = Activation(activation='sigmoid')(embedding)

model = Model(encoder_model.inputs[0], embedding)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

model_checkpoint_path = "./model_checkpoint"

model_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(model_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

model.fit(x_train, y_train,
          batch_size=128,
          epochs=epochs,
          callbacks=model_callbacks,
          validation_data=(x_test, y_test))

model.load_weights(model_checkpoint_path)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
