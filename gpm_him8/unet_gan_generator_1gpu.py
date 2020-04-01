from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf

import os
import time

def Generator():
    concat_axis = 3
    inputs = layers.Input(shape = (512, 512, 2))

    feats = 4#16
    bn0 = layers.BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn2 = layers.BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2) #256

    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn4 = layers.BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4) #128

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn6 = layers.BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6) #64

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn8 = layers.BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8) #32

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn10 = layers.BatchNormalization(axis=3)(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10) #16

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn11) #32
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)

    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = layers.BatchNormalization(axis=3)(conv7)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn13) #64
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)

    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn15) #128
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = layers.BatchNormalization(axis=3)(conv7)

    # Rectify last convolution layer to constraint output to positive precipitation values.
    conv8 = layers.Conv2D(1, (1, 1), activation='relu')(bn13)

    model = models.Model(inputs=inputs, outputs=conv8)


    return model


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')
  inp = tf.keras.layers.Input(shape=[None, None, 2], name='input_image')

  x = tf.keras.layers.concatenate([inp, tar])

  down1 = downsample(64, 3, False)(x)
  down2 = downsample(128, 3)(down1)
  down3 = downsample(256, 3)(down2)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(512, 3, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  #total_gen_loss = l1_loss

  return total_gen_loss


generator = Generator()
#generator= tf.keras.models.load_model('gan_generator.h5')
discriminator = Discriminator()
#discriminator = tf.keras.models.load_model('gan_discriminator.h5')
generator_optimizer = tf.keras.optimizers.Adagrad(lr=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adagrad(lr=0.0001)


@tf.function
def train_step(input_image, target):
  target_masked = tf.where(tf.math.is_nan(target), tf.zeros_like(target), target)

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)
    gen_output_masked = tf.where(tf.math.is_nan(target), tf.zeros_like(target), gen_output)
    
    disc_real_output = discriminator([input_image, target_masked], training=True)
    disc_generated_output = discriminator([input_image, gen_output_masked], training=True)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_loss = generator_loss(disc_generated_output, gen_output, target)

  generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

import numpy as np

def fit(train_ds, test_ds, epochs):
  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  template = 'Epoch {}, Loss: {:.4f}, Test Loss: {:.4f}\n'

  f = open("train_record.out","w+")

  for epoch in range(epochs):
    start = time.time()

    # Train
    for batch, (inputs, target) in enumerate(train_ds):
      train_step(inputs, target)
      train_loss(calc_loss(generator, inputs, target))
    
    for batch, (inputs, target) in enumerate(test_ds):
      test_loss(calc_loss(generator, inputs, target))

    print(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.write(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.flush() 
   
    train_loss.reset_states()
    test_loss.reset_states()

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

  f.close()
  generator.save('gan_generator.h5')
  discriminator.save('gan_discriminator.h5')

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

print(y_train.max(), y_test.max())

y_train = np.clip(y_train,0,30)
y_test = np.clip(y_test,0,30)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
fit(train_dataset, test_dataset, 100)

