import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import data_utils as du

def build_and_compile_model():
    inputs = layers.Input(shape=(28,))
    x = layers.Dense(512,activation='relu')(inputs)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dense(512,activation='relu')(x)
    outputs = layers.Dense(14)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=optimizers.Adam(0.001))
    return model

dataset = du.load_tfrecords()
dataset = dataset.batch(64, drop_remainder=False)

model = build_and_compile_model()
model.summary()

model.fit(dataset, verbose=1, epochs=200)

model.save('./model/simple/simple.h5')
model = models.load_model('./model/simple/simple.h5')

model.predict(dataset)