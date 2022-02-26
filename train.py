import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization
import data_utils as du


def build_and_compile_model():
    model = Sequential([
        InputLayer(input_shape=(None,28,)),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(14)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dataset = du.load_tfrecords()

dataset = dataset.batch(1024)

model = build_and_compile_model()
model.summary()

model.fit(dataset, verbose=1, epochs=50)

model.save('./model/simple/')