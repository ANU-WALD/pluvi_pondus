from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf

import os
import time

from nc_loader_small import HimfieldsDataset

def Unet():
    concat_axis = 3
    #inputs = layers.Input(shape = (256, 256, 2))
    inputs = layers.Input(shape = (1024, 1024, 4))
    #inputs = layers.Input(shape = (2048,2048,4))

    feats = 16
    bn0 = layers.BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = layers.BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = layers.BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)
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

    conv10 = layers.Conv2D(1, (1, 1))(bn18)
    #bn19 = BatchNormalization(axis=3)(conv10)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)


    return model

def mse_holes(y_true, y_pred):
    #idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

def mse_holes(y_true, y_pred):
    #idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)


@tf.function
def train_step(model, inputs, outputs, optimizer):

  with tf.GradientTape() as t:
    loss = mse_holes(outputs, model(inputs, training=True))

    grads = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


@tf.function
def calc_loss(model, inputs, outputs):
    return mse_holes(outputs, model(inputs))


def train(train_dataset, test_dataset, epochs):
  model = Unet()
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  
  train_loss = tf.keras.metrics.Mean()
  test_loss = tf.keras.metrics.Mean()
  template = 'Epoch {}, Loss: {:.4f}, Test Loss: {:.4f}\n'

  f = open("train_record_unet.out","w+")

  for epoch in range(epochs):

    for (batch, (inputs, outputs)) in enumerate(train_dataset):
      train_step(model, inputs, outputs, optimizer)
      train_loss(calc_loss(model, inputs, outputs))

    for (inputs, outputs) in test_dataset:
      test_loss(calc_loss(model, inputs, outputs))
    
    print(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.write(template.format(epoch+1, train_loss.result(), test_loss.result()))
    f.flush() 

    train_loss.reset_states()
    test_loss.reset_states()

  f.close()
  model.save('unet.h5')


"""
train_fnames = ["/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181101.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181102.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181103.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181104.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181105.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181106.nc"]
"""

train_fnames = ["/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181101.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181102.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181103.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181104.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181105.nc",
                "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_20181106.nc"]

train_dataset = HimfieldsDataset(train_fnames, 1, batch_size=4)
test_dataset = HimfieldsDataset(train_fnames, 1, batch_size=4)
EPOCHS = 40
train(train_dataset, test_dataset, EPOCHS)