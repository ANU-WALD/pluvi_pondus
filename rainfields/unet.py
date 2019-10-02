from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as K
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from nc_loader import HimfieldsDataset


def mse_holes(y_true, y_pred):
    #idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)


def downsample(filters, size):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', activation='relu'))
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', activation='relu'))
  result.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

  return result


def upsample(filters, size, last=False):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', activation='relu'))
  #result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', activation='relu'))
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', activation='relu'))
  if last:
    pass
    result.add(tf.keras.layers.BatchNormalization(axis=3))
  return result

def get_unet():
    concat_axis = 3
    inputs = tf.keras.layers.Input(shape = (2048,2048,2))

    feats = 3
    bn0 = tf.keras.layers.BatchNormalization(axis=3)(inputs)
    conv1 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    conv1 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)

    conv2 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    conv2 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn4)
    
    conv3 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = tf.keras.layers.BatchNormalization(axis=3)(conv3)
    conv3 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = tf.keras.layers.BatchNormalization(axis=3)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn6)
    
    conv4 = tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn6 = tf.keras.layers.BatchNormalization(axis=3)(conv4)
    conv4 = tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn6)
    bn7 = tf.keras.layers.BatchNormalization(axis=3)(conv4)

    up_conv7 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn7)
    up8 = tf.keras.layers.concatenate([up_conv7, conv3], axis=concat_axis)
    conv8 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = tf.keras.layers.BatchNormalization(axis=3)(conv8)
    conv8 = tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = tf.keras.layers.BatchNormalization(axis=3)(conv8)
    
    up_conv7 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn16)
    up8 = tf.keras.layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = tf.keras.layers.BatchNormalization(axis=3)(conv8)
    conv8 = tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = tf.keras.layers.BatchNormalization(axis=3)(conv8)

    up_conv8 = tf.keras.layers.UpSampling2D(size=(2, 2))(bn16)
    up9 = tf.keras.layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(up9)
    bn17 = tf.keras.layers.BatchNormalization(axis=3)(conv9)
    conv9 = tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = tf.keras.layers.BatchNormalization(axis=3)(conv9)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(bn18)
    #bn19 = BatchNormalization(axis=3)(conv10)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    print(model.summary())

    return model

def Unet():
  nf = 8
  down_stack = [
    downsample(nf, 3), #1024
    downsample(nf*2, 3), #512
    #downsample(nf*4, 3), #256
    #downsample(nf*8, 3), #128
    #downsample(nf*16, 3), #64
  ]

  up_stack = [
    #upsample(nf*16, 3),
    #upsample(nf*8, 3),
    #upsample(nf*4, 3),
    upsample(nf*2, 3),
    upsample(nf, 3, last=True),
  ]

  last = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu')

  inputs = tf.keras.layers.Input(shape=[None,None,2])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  concat = tf.keras.layers.Concatenate()
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def custom_loss(gen_output, target):
  # mean square error
  return mse_holes(gen_output, target)


train_fnames = ["/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181101.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181102.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181103.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181104.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181105.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181106.nc"]

for n in [1]:
  #model = Unet()
  model = get_unet()
  print(model.summary())
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss=custom_loss)
  train_dataset = HimfieldsDataset(train_fnames, n, batch_size=4)
  test_dataset = HimfieldsDataset(train_fnames, n, batch_size=4)
  model.fit_generator(train_dataset, epochs=10, verbose=1, validation_data=test_dataset)
