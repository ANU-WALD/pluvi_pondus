from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

x = np.load("x_conv.npy")[:10000000]
print(x.shape)

y = np.load("y_conv.npy")[:10000000,None]
print(y.shape)

prec_mask = np.nonzero(y>0)
print(prec_mask)
print(len(prec_mask))
#print(prec_mask.shape)
#print(prec_mask[0])
print(prec_mask[0].shape)

x_prec = x[prec_mask[0], :]
y_prec = y[prec_mask[0], :]
print(x_prec.shape, y_prec.shape)

zero_mask = np.nonzero(y==0)
x_dry = x[zero_mask[0], :]
y_dry = y[zero_mask[0], :]
print(x_dry.shape, y_dry.shape)

idxs = np.arange(x_dry.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)
n = x_prec.shape[0] * 2
x_dry = x_dry[idxs[:n],:]
y_dry = y_dry[idxs[:n],:]
print(x_dry.shape, y_dry.shape)

x = np.concatenate((x_prec, x_dry), axis=0)
y = np.concatenate((y_prec, y_dry), axis=0)
print(x.shape, y.shape)

idxs = np.arange(x.shape[0])
np.random.shuffle(idxs)

x = x[idxs,:]
x = np.reshape(x, (x.shape[0], -1))
y = y[idxs,:]
print(x.shape, y.shape)

model= Sequential()
model.add(Dense(100, activation='relu', input_dim=100))
model.add(Dense(200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

"""
model= Sequential()
model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape=(5,5,4)))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
"""

x_train = x[:175000,:]
x_test = x[175000:,:]
y_train = y[:175000,:]
y_test = y[175000:,:]
print(y_train.shape, y_test.shape)
print(np.square(y_train).mean(axis=0))
print(np.square(y_test).mean(axis=0))
print(np.abs(y_train).mean(axis=0))
print(np.abs(y_test).mean(axis=0))

#classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr=0.000001), loss='mse', metrics=['mae', 'mse'])

model.fit(x_train, y_train, batch_size=32, nb_epoch=10, validation_data=(x_test, y_test))
