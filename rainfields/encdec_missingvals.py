from keras import layers
from keras import models
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import Sequence
import numpy as np
from keras import backend as K
import tensorflow as tf
import xarray as xr
from datetime import datetime
from datetime import timedelta
import random
import os

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
    idxs = tf.where(tf.math.logical_not(tf.math.is_nan(y_true)))
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

class DataGenerator(Sequence):
    def __init__(self, batch_size=4, length=40):
        'Initialization'
        self.batch_size = batch_size
        self.length = length

    def __len__(self):
        return int(self.length/self.batch_size)

    def __getitem__(self, index):
        x = []
        y = []

        while len(y) < self.batch_size:
            n = random.randint(1,6*24*6)
            d = datetime(2018,11,1,0,0) + timedelta(0,10*60*n)
            dp = d - timedelta(0,10*60)
            rf_fp = "/home/lar116/project/pablo/rainfields_data/310_{}_{}.prcp-c10.npy".format(d.strftime("%Y%m%d"), d.strftime("%H%M%S"))
            h8_fp = "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(d.strftime("%Y%m%d"))
            h8p_fp = "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(dp.strftime("%Y%m%d"))
            
            if not os.path.exists(rf_fp) or not os.path.exists(h8_fp) or not os.path.exists(h8p_fp):
                continue
           
            h8_ds = xr.open_dataset(h8_fp)
            h8p_ds = xr.open_dataset(h8p_fp)

            if np.datetime64(d) not in h8_ds.time.data or np.datetime64(dp) not in h8p_ds.time.data:
                continue

            prec = np.load(rf_fp)[2:, 402:]
            b8 = h8_ds.B8.sel(time=d)[2:, 402:].data
            b14 = h8_ds.B14.sel(time=d)[2:, 402:].data
            b8p = h8p_ds.B8.sel(time=dp)[2:, 402:].data
            b14p = h8p_ds.B14.sel(time=dp)[2:, 402:].data

            x.append(np.stack((b8p,b14p,b8,b14), axis=-1))
            #x.append(np.stack((b8,b14), axis=-1))
            y.append(prec)

        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)[:,:,:,None]

        return x, y

    def on_epoch_end(self):
        pass


def get_unet():
    concat_axis = 3
    #inputs = layers.Input(shape = (1024, 1024, 4))
    inputs = layers.Input(shape = (2048,2048,4))

    feats = 8
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

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=mse_holes, optimizer=sgd)
    print(model.summary())

    return model

training_gen = DataGenerator(batch_size=4, length=400)
validation_gen = DataGenerator(batch_size=4, length=100)

model = get_unet()
print(get_model_memory_usage(1, model), "GBs")
#exit()

#history = model.fit(x_train, y_train, epochs=50, batch_size=4, validation_data=(x_test, y_test))
#history = model.fit_generator(generator=training_gen, validation_data=validation_gen, use_multiprocessing=True, workers=2)
history = model.fit_generator(generator=training_gen, validation_data=validation_gen, epochs=100, max_queue_size=8, use_multiprocessing=True, workers=4)
model.save('rainfields_model.h5')
