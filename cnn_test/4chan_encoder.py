from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
import numpy as np

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(80, 120, 1)))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same'))

    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['mse'])
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001), metrics=['mae'])

    print(model.summary())

    return model

x = np.expand_dims(np.load("/scratch/director2107/ERA5_Data/ERA5_HIM8/era5_z.npy"), axis=3)
y = 1000*np.expand_dims(np.load("/scratch/director2107/ERA5_Data/ERA5_HIM8/era5_tp.npy"), axis=3)

print(x.shape)
print(y.shape)

for i in range(10):
    x_train = x[:9000, :]
    y_train = y[:9000, :]

    x_test = x[9000:, :]
    y_test = y[9000:, :]

    model = GetModel()
    history = model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))
