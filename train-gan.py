import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model, Sequential
import time
import data_utils as du

# def build_and_compile_model():
#     model = Sequential([
#         Input(shape=(28,)),
#         Dense(1024, activation='relu'),
#         Dense(512, activation='relu'),
#         Dense(1024, activation='relu'),
#         Dense(14)
#     ])
#     model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
#     return model


def make_generator_model()->Model:
    model = Sequential()
    model.add(layers.Input(shape=(28,)))
    model.add(layers.Dense(128,use_bias=False))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(128, use_bias=False))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(14, use_bias=False))
    model.add(layers.Activation(activations.tanh))
    return model

def make_discriminator_model()->Model:
    model = Sequential()
    model.add(layers.Input(shape=(14,)))
    model.add(layers.Dense(128))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(128))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(1))
    return model

dataset = du.load_tfrecords2D()

dataset = dataset.batch(8)

generator = make_generator_model()
discriminator = make_discriminator_model()

data = np.array(list(dataset.take(1).as_numpy_iterator())[0])

print(generator(data,training=False))