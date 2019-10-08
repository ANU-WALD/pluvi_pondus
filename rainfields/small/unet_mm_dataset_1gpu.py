from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import SGD, Adam
import os
import time
import matplotlib.pyplot as plt
from nc_loader_small import HimfieldsDataset


def mse_holes(y_true, y_pred): #idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)


def get_unet():
    concat_axis = 3
    inputs = tf.keras.layers.Input(shape = (1024, 1024, 4))

    feats = 16

    model = tf.keras.Sequential()    
    # Encoder

    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3)) #1024
   
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #512
    model.add(tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #256
    model.add(tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #128
    model.add(tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #64
    model.add(tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #32
    model.add(tf.keras.layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))

    model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) #64

    # Decoder
    
    
    model.add(tf.keras.layers.concatenate([up_conv1.output, conv5.output], axis=concat_axis))

    model.add(tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) #128
 
    model.add(tf.keras.layers.concatenate([up_conv2, conv4], axis=concat_axis))

    model.add(tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) #256
    
    model.add(tf.keras.layers.concatenate([up_conv3, conv3], axis=concat_axis))

    model.add(tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) #512
    
    model.add(tf.keras.layers.concatenate([up_conv4, conv2], axis=concat_axis))

    model.add(tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) #1024
    
    model.add(tf.keras.layers.concatenate([up_conv5, conv1], axis=concat_axis))

    model.add(tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.Conv2D(feats, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
 
    model.add(tf.keras.layers.Conv2D(1, (1, 1), activation='relu'))
    #bn19 = BatchNormalization(axis=3)(conv10)

    return model


def downsample(filters, first=False, last=False):

  result = tf.keras.Sequential()

  if first:
    result.add(tf.keras.layers.BatchNormalization(axis=3))

  result.add(tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', activation='relu'))
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  result.add(tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', activation='relu'))
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  if not last:
    result.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

  return result


def upsample(filters, first=False, last=False):

  result = tf.keras.Sequential()
  
  result.add(tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', activation='relu'))
  result.add(tf.keras.layers.BatchNormalization(axis=3))
  result.add(tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', activation='relu'))
  result.add(tf.keras.layers.BatchNormalization(axis=3))

  return result


def Unet():
  feats = 8
  down_stack = [
    downsample(feats, first=True), #512
    downsample(feats*2), #256
    downsample(feats*4), #128
    downsample(feats*8), #64
    downsample(feats*16), #32
    downsample(feats*32, last=True), #32
  ]

  up_stack = [
    upsample(feats*16), #32
    upsample(feats*8), #64
    upsample(feats*4), #128
    upsample(feats*2), #256
    upsample(feats), #512
  ]
  
  upper = tf.keras.layers.UpSampling2D(size=(2, 2))
  last = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='relu')

  inputs = tf.keras.layers.Input(shape=[1024,1024,4])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-2])

  concat = tf.keras.layers.Concatenate()
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    #x = up(x)
    x = concat([tf.keras.layers.UpSampling2D(size=(2, 2))(x), skip])
    #x = concat([x, skip])

  x = last(upper(x))

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

train_fnames = ["/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181101.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181102.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181103.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181104.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181105.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181106.nc"]

training_gen = HimfieldsDataset(train_fnames, 1, batch_size=2)
validation_gen = HimfieldsDataset(train_fnames, 1, batch_size=2)
  
#model = get_unet()
model = Unet()
print(model.summary())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=mse_holes, optimizer=sgd)
history = model.fit_generator(generator=training_gen, validation_data=validation_gen, verbose=1, epochs=100)
exit()

with open('train_history_him8_8batch.pkl', 'wb') as f:
    pickle.dump(history.history, f)

parallel_model.save('rainfields_model_mse_small.h5')
"""
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss=custom_loss)
  train_dataset = HimfieldsDataset(train_fnames, n, batch_size=2)
  test_dataset = HimfieldsDataset(train_fnames, n, batch_size=2)
  model.fit_generator(train_dataset, epochs=100, verbose=1, validation_data=test_dataset)
"""
