from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Add, Activation
import time
import data_utils as du
import random
import tensorflow as tf
import datetime


def layer(x: tf.Tensor, hidden: int) -> tf.Tensor:
    x = Dense(hidden, kernel_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return Dropout(0.5)(x)

def residual_block(x: tf.Tensor, hidden: int) -> tf.Tensor:
    y = layer(x, hidden)
    y = layer(y, hidden)
    return Add()([x,y])


def make_generator_model()->Model:
    hidden = 1024
    inputs = Input(shape=(28,))
    x = layer(inputs, hidden)
    x = residual_block(x,hidden)
    x = residual_block(x,hidden)
    outputs = Dense(14)(x)
    return Model(inputs=inputs, outputs=outputs)

def make_discriminator_model()->Model:
    hidden = 1024
    inputs = Input(shape=(28,))
    x = layer(inputs, hidden)
    x = residual_block(x,hidden)
    x = residual_block(x,hidden)
    outputs = Dense(1)(x)
    return Model(inputs=inputs, outputs=outputs)

generator = make_generator_model()
generator.summary()
discriminator = make_discriminator_model()
discriminator.summary()


EPOCHS = 200
MIN_EPOCHS = 100
batch_size=64
dataset = du.load_tfrecords2D().shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
CW = tf.convert_to_tensor(du.camera_to_world(),dtype=np.float32)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

tensorboard_gen_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
tensorboard_disc_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)

@tf.function
def train_step(X):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        Y = generator(X,training=True)
        X = tf.reshape(X,(-1, 14, 2))
        ones = tf.ones(Y.shape);
        body = tf.stack([X[:,:,0],X[:,:,1], Y, ones], axis=2)
        body_3d = tf.transpose(tf.matmul(CW,body,transpose_a=False,transpose_b=True),perm=[0,2,1])
        azimuth = random.uniform(0, 2*np.pi)
        altitude = 0
        WC = tf.convert_to_tensor(du.world_to_camera(azimuth, altitude),dtype=np.float32)
        body_camera_rot = tf.transpose(tf.matmul(WC,body_3d,transpose_a=False,transpose_b=True),perm=[0,2,1])
        
        real_output = discriminator(tf.reshape(X,(-1,28)), training=True)
        fake_output = discriminator(tf.reshape(body_camera_rot[:,:,0:2],(-1,28)), training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    tensorboard_gen_loss(gen_loss)
    tensorboard_disc_loss(disc_loss)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/gradient_tape/' + current_time + '/gan'
summary_writer = tf.summary.create_file_writer(log_dir)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for batch in dataset:
            train_step(batch)
        with summary_writer.as_default():
            tf.summary.scalar('gen_loss', tensorboard_gen_loss.result(), step=epoch)
            tf.summary.scalar('disc_loss', tensorboard_disc_loss.result(), step=epoch)

        template = 'Epoch {} - {:.2f} sec, Gen Loss: {}, Disc Loss: {}'
        print(template.format(epoch+1, time.time()-start, tensorboard_gen_loss.result(), tensorboard_disc_loss.result()))

        if np.abs(tensorboard_disc_loss.result()-1)<0.1 and epoch+1 >= MIN_EPOCHS:
            print('Early termination {}'.format(epoch+1))
            break;

        # Reset metrics every epoch
        tensorboard_gen_loss.reset_states()
        tensorboard_disc_loss.reset_states()
        model_fname = './model/gan/gan_{}.h5'.format(epoch+1)
        generator.save(model_fname)



train(dataset, EPOCHS)

model_fname = './model/gan/gan.h5'
generator.save(model_fname)

