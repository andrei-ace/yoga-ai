import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Add, Activation
import data_utils as du

def build_and_compile_model():
    model = Sequential([
        Input(shape=(28,)),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(14)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dataset = du.load_tfrecords()

dataset = dataset.batch(512)

model = build_and_compile_model()
model.summary()

model.fit(dataset, verbose=1, epochs=100)

model.save('./model/simple/')