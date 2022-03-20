import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Add, Activation
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
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


batch_size = 64
dataset = du.load_tfrecords().shuffle(1000*batch_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size)

model_fname = './model/article/article.h5'
if os.path.isfile(model_fname):
    model = load_model(model_fname)
else:
    model = build_and_compile_model()

model.summary()

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.0001)
model.fit(dataset, verbose=1, epochs=200, callbacks=[reduce_lr])

if os.path.isfile(model_fname):
    os.rename(model_fname, model_fname+".bak")

model.save(model_fname)
