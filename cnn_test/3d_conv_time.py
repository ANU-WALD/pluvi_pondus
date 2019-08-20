from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv3D, Conv3DTranspose
from tensorflow.keras.optimizers import Adam
import sys
import numpy as np

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(2, 80, 120, 1)))
    model.add(Conv3D(32, (2, 5, 5), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3D(32, (2, 5, 5), strides=(1, 2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3D(128, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3D(128, (3, 3, 3), strides=(1, 2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3D(256, (3, 5, 5), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3D(256, (3, 5, 5), strides=(1, 2, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3DTranspose(128, (3, 5, 5), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3DTranspose(128, (3, 5, 5), strides=(1, 2, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3DTranspose(64, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3DTranspose(64, (3, 3, 3), strides=(1, 2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3DTranspose(32, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3DTranspose(32, (3, 3, 3), strides=(1, 2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv3DTranspose(1, (2, 5, 5), strides=(1, 1, 1), activation='relu', padding='same'))
    model.add(Conv3DTranspose(1, (2, 5, 5), strides=(1, 2, 2), activation='relu', padding='same'))
    model.add(Conv3D(1, (1, 1, 1), strides=(2, 1, 1), activation='relu', padding='same'))

    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['mse'])
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.005), metrics=['mae'])

    print(model.summary())

    return model

#him8 = np.load("/home/pl5189/ERA5_HIM8_CNN/him8_b8.npy")
#np.save("/home/pl5189/ERA5_HIM8_CNN/era5_z_aux.npy", x)
#x = np.load("/home/pl5189/ERA5_HIM8_CNN/era5_z_rec.npy")[:,:,:,2,None].copy()

#era5 = np.load("/scratch/director2107/ERA5_Data/ERA5_HIM8/era5_z_aux.npy")[:,:,:,0]
#x = np.load("/scratch/director2107/ERA5_Data/ERA5_HIM8/era5_z_aux.npy")[:]
z500 = np.load("/scratch/director2107/ERA5_Data/ERA5_HIM8/500_geopotential.npy")
print(z500[:-2,:] - z500[1:-1,:])
print(np.mean(z500[:-2,:] - z500[1:-1,:]))
print(np.mean(np.abs(z500[:-1] - z500[1:])))

x = np.stack([z500[:-2,:], z500[1:-1,:]], axis=1)[:, :, :, :, None]
y = z500[2:, :][:, None, :, :, None]

idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

y = y[idxs, :]
x = x[idxs, :]

y_train = y[:40000, :]
y_test = y[40000:, :]

x_train = x[:40000, :]
x_test = x[40000:, :]

model = GetModel()
history = model.fit(x_train, y_train, batch_size=24, epochs=40, verbose=1, validation_data=(x_test, y_test))
