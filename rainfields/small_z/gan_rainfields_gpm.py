from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf

import os
import time

import matplotlib.pyplot as plt

from data_loader_2chan_rainfields import HimfieldsDataset

def Generator():
    concat_axis = 3
    ref_input = layers.Input(shape = (1024, 1024, 2))
    gpm_input = layers.Input(shape = (512, 512, 1))

    feats = 16
    bn0 = layers.BatchNormalization(axis=3)(ref_input)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = layers.BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)

    gpmadd = layers.concatenate([gpm_input, pool1], axis=concat_axis)
    gpmaddn = layers.BatchNormalization(axis=3)(gpmadd)
    
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = layers.BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = layers.BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = layers.BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = layers.BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6)

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = layers.BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = layers.BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8)

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = layers.BatchNormalization(axis=3)(conv5)
    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = layers.BatchNormalization(axis=3)(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10)

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = layers.BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12)
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)

    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = layers.BatchNormalization(axis=3)(conv6)
    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn12)
    bn14 = layers.BatchNormalization(axis=3)(conv6)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn10)
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)

    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = layers.BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = layers.BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn16)
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = layers.BatchNormalization(axis=3)(conv7)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = layers.BatchNormalization(axis=3)(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(bn14)
    up8 = layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = layers.BatchNormalization(axis=3)(conv8)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = layers.BatchNormalization(axis=3)(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(bn16)
    up9 = layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(up9)
    bn17 = layers.BatchNormalization(axis=3)(conv9)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = layers.BatchNormalization(axis=3)(conv9)

    conv10 = layers.Conv2D(1, (1, 1), activation='relu')(bn18)
    #bn19 = BatchNormalization(axis=3)(conv10)

    model = tf.keras.models.Model(inputs=[ref_input,gpm_input], outputs=conv10)

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

  down1 = downsample(64, 4, False)(x)
  down2 = downsample(128, 4)(down1)
  down3 = downsample(256, 4)(down2)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
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
  #l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  l1_loss = mae_holes(target, gen_output)
  #l2_loss = mse_holes(target, gen_output)

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  #total_gen_loss = l1_loss

  return total_gen_loss


generator = Generator()
#generator= tf.keras.models.load_model('gan_2chanrc_generator.h5')
discriminator = Discriminator()
#discriminator = tf.keras.models.load_model('gan_2chanrc_discriminator.h5')
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def mse_holes(y_true, y_pred):
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

def mae_holes(y_true, y_pred):
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.abs(y_true-y_pred), axis=-1)

@tf.function
def calc_loss(model, inputs, outputs):
    return mse_holes(outputs, model(inputs))

@tf.function
def train_step(input_ref, input_gpm, target):
  target_masked = tf.where(tf.math.is_nan(target), tf.zeros_like(target), target)

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator([input_ref, input_gpm], training=True)
    gen_output_masked = tf.where(tf.math.is_nan(target), tf.zeros_like(target), gen_output)
    
    disc_real_output = discriminator([input_ref, target_masked], training=True)
    disc_generated_output = discriminator([input_ref, gen_output_masked], training=True)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_loss = generator_loss(disc_generated_output, gen_output, target)

  generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


def fit(train_ds, test_ds, epochs):
  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  template = 'Epoch {}, Loss: {:.4f}, Test Loss: {:.4f}\n'

  f = open("train_record_gan_mse_rainfields_gpm.out","w+")

  for epoch in range(epochs):
    start = time.time()

    # Train
    for batch, (ref_input, gpm_input, target) in enumerate(train_ds):
      train_step(ref_input, gpm_input, target)
      train_loss(calc_loss(generator, [ref_input, gpm_input], target))
    
    for batch, (ref_input, gpm_input, target) in enumerate(test_ds):
      test_loss(calc_loss(generator, [ref_input, gpm_input], target))

    print(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.write(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.flush() 
   
    train_loss.reset_states()
    test_loss.reset_states()

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

  f.close()
  generator.save('gan_mse_rainfields_gpm_generator.h5')
  discriminator.save('gan_mse_rainfields_gpm_discriminator.h5')


train_fnames = ["/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181101.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181102.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181103.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181104.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181105.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181106.nc"]

test_fnames = ["/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181107.nc",
               "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181108.nc",
               "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181109.nc",
               "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181110.nc",
               "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181111.nc",
               "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181112.nc"]


train_dataset = HimfieldsDataset(train_fnames, "gpm", batch_size=4)
test_dataset = HimfieldsDataset(test_fnames, "gpm", batch_size=4)

EPOCHS = 50
fit(train_dataset, test_dataset, EPOCHS)

