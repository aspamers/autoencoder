# Auto-Encoder for Keras

This project provides a lightweight, easy to use and flexible auto-encoder module for use with the Keras 
framework. 

Auto-encoders are used to generate embeddings that describe inter and extra class relationships. 
This makes auto-encoders like many other similarity learning algorithms suitable as a pre-training step for many 
classification problems. 

An example of the auto-encoder module being used to produce a noteworthy 99.84% validation performance on the MNIST 
dataset with no data augmentation and minimal modification from the Keras example is provided.

## Installation

Create and activate a virtual environment for the project.
```sh
$ virtualenv env
$ source env/bin/activate
```

To install the module directly from GitHub:
```
$ pip install git+https://github.com/aspamers/autoencoder
```

The module will install keras and numpy but no back-end (like tensorflow). This is deliberate since it leaves the module 
decoupled from any back-end and gives you a chance to install whatever version you prefer. To install tensorflow:
```
$ pip install tensorflow
```

## Usage
For detailed usage examples please refer to the examples and unit test modules. If the instructions are not sufficient 
feel free to make a request for improvements.

- Import the module
```python
from autoencoder import AutoEncoder
```

- Load or generate some data.
```python
x_train = np.random.rand(100, 3)
x_test = np.random.rand(30, 3)
```

- Design an encoder model
```python
def create_encoder_model(input_shape):
    model_input = Input(shape=input_shape)

    encoder = Dense(4)(model_input)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(activation='relu')(encoder)

    return Model(model_input, encoder)
```

- Design a decoder model
```python

    def create_decoder_model(embedding_shape):
        embedding_a = Input(shape=embedding_shape)

        decoder = Dense(3)(embedding_a)
        decoder = BatchNormalization()(decoder)
        decoder = Activation(activation='relu')(decoder)

        return Model(embedding_a, decoder)
```

- Create an instance of the AutoEncoder class
```python
encoder_model = create_encoder_model(input_shape)
decoder_model = create_decoder_model(encoder_model.output_shape)
autoencoder = AutoEncoder(encoder_model, decoder_model)
```

- Compile the model
```python
autoencoder.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam())
```

- Train the model
```python
autoencoder.fit(x_train, x_train,
                validation_data=(x_test, x_test),
                epochs=epochs)
```

## Development Environment
Create and activate a test virtual environment for the project.
```sh
$ virtualenv env
$ source env/bin/activate
```

Install requirements
```sh
$ pip install -r requirements.txt
```

Run tests
```sh
$ pytest tests/test_autoencoder.py
```