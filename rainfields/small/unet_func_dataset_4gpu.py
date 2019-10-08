import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
#from keras.utils import Sequence
import numpy as np
from tensorflow.keras import backend as K
import xarray as xr
from datetime import datetime
from datetime import timedelta
import pickle
import os
from nc_loader_small import HimfieldsDataset

def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        print(l.name)
        print(l.output_shape)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def mse_holes(y_true, y_pred):
    #idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

def get_unet():
    concat_axis = 3
    #inputs = layers.Input(shape = (256, 256, 2))
    inputs = layers.Input(shape = (1024, 1024, 4))
    #inputs = layers.Input(shape = (2048,2048,4))

    feats = 16
    bn0 = BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6)

    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(bn8)

    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = BatchNormalization(axis=3)(conv5)
    conv5 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = BatchNormalization(axis=3)(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(bn10)

    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(pool5)
    bn11 = BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(32*feats, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12)
    up7 = layers.concatenate([up_conv6, conv5], axis=concat_axis)
    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv6)
    conv7 = layers.Conv2D(16*feats, (3, 3), activation='relu', padding='same')(bn12)
    bn14 = BatchNormalization(axis=3)(conv6)
    
    up_conv5 = layers.UpSampling2D(size=(2, 2))(bn10)
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(up6)
    bn15 = BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(8*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn16)
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)
    conv7 = layers.Conv2D(4*feats, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = BatchNormalization(axis=3)(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(bn14)
    up8 = layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(up8)
    bn15 = BatchNormalization(axis=3)(conv8)
    conv8 = layers.Conv2D(2*feats, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = BatchNormalization(axis=3)(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(bn16)
    up9 = layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(up9)
    bn17 = BatchNormalization(axis=3)(conv9)
    conv9 = layers.Conv2D(feats, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = BatchNormalization(axis=3)(conv9)

    conv10 = layers.Conv2D(1, (1, 1))(bn18)
    #bn19 = BatchNormalization(axis=3)(conv10)

    model = models.Model(inputs=inputs, outputs=conv10)


    return model

train_fnames = ["/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181101.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181102.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181103.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181104.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181105.nc",
                "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_20181106.nc"]

training_gen = HimfieldsDataset(train_fnames, 1, batch_size=8)
validation_gen = HimfieldsDataset(train_fnames, 1, batch_size=8)

model = get_unet()
print(model.summary())
#print(get_model_memory_usage(8, model), "GBs")
parallel_model = multi_gpu_model(model, gpus=4)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
parallel_model.compile(loss=mse_holes, optimizer=sgd)
#model.compile(loss=mse_holes, optimizer=sgd)

#history = model.fit_generator(generator=training_gen, validation_data=validation_gen, use_multiprocessing=True, workers=2)
history = parallel_model.fit_generator(generator=training_gen, validation_data=validation_gen, epochs=100)
#history = model.fit_generator(generator=training_gen, validation_data=validation_gen, epochs=100, max_queue_size=8, use_multiprocessing=True, verbose=1, workers=8)
#history = model.fit_generator(generator=training_gen, validation_data=validation_gen, epochs=100, verbose=1)
exit()

with open('train_history_him8_8batch.pkl', 'wb') as f:
    pickle.dump(history.history, f)

parallel_model.save('rainfields_model_mse_small.h5')
