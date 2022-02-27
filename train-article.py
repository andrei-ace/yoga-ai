import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Add, Activation
from tensorflow.keras.constraints import max_norm
import data_utils as du


def layer(x: tf.Tensor, hidden: int) -> tf.Tensor:
    x = Dense(hidden, kernel_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return Dropout(0.5)(x)

def residual_block(x: tf.Tensor, hidden: int) -> tf.Tensor:
    y = layer(x, hidden)
    y = layer(y, hidden)
    return Add()([x,y])

def build_and_compile_model():
    hidden = 1024
    inputs = Input(shape=(28,))
    x = layer(inputs, hidden)
    x = residual_block(x,hidden)
    x = residual_block(x,hidden)
    outputs = Dense(14)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dataset = du.load_tfrecords()

dataset = dataset.batch(64)

model = build_and_compile_model()
model.summary()

model.fit(dataset, verbose=1, epochs=200)

model.save('./model/article/')