from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
import sys
import numpy as np

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(240, 360, 2)))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (5, 5), strides=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(128, (5, 5), strides=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same'))

    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['mse'])
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001), metrics=['mae'])

    print(model.summary())

    return model

#x = np.moveaxis(np.load("/home/pl5189/ERA5_HIM8_CNN/era5_z.npy"), 1, -1)[:,:240,:360,:]
#np.save("/home/pl5189/ERA5_HIM8_CNN/era5_z_rec.npy", x)
#him8 = np.load("/home/pl5189/ERA5_HIM8_CNN/him8_b8.npy")
#np.save("/home/pl5189/ERA5_HIM8_CNN/era5_z_aux.npy", x)
#x = np.load("/home/pl5189/ERA5_HIM8_CNN/era5_z_rec.npy")[:,:,:,2,None].copy()
era5 = np.load("/home/pl5189/ERA5_HIM8_CNN/era5_z_aux.npy")[:]
him8 = np.load("/home/pl5189/ERA5_HIM8_CNN/him8_b7.npy")[1:, :, :, None]
x = np.stack([him8, era5], axis=3)
print(x.shape)
era5 = None
him8 = None
#y = np.load("/home/pl5189/ERA5_HIM8_CNN/era5_tp_rec.npy")[:26338,:]
y = np.load("/home/pl5189/ERA5_HIM8_CNN/era5_tp_rec.npy")[1:,:]



x_train = x[:20000, :]
y_train = y[:20000, :]

x_test = x[20000:, :]
y_test = y[20000:, :]

model = GetModel()
history = model.fit(x_train, y_train, batch_size=24, epochs=10, verbose=1, validation_data=(x_test, y_test))
