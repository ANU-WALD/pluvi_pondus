from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

x = np.load("x.npy")[:10000000]
print(x.shape)

y = np.load("y.npy")[:10000000,None]
print(y.shape)

prec_mask = np.nonzero(y>0)
x_prec = x[prec_mask[0], :]
y_prec = y[prec_mask[0], :]

zero_mask = np.nonzero(y==0)
x_dry = x[zero_mask[0], :]
y_dry = y[zero_mask[0], :]

idxs = np.arange(x_dry.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)
n = x_prec.shape[0] * 2
x_dry = x_dry[idxs[:n],:]
y_dry = y_dry[idxs[:n],:]

x = np.concatenate((x_prec, x_dry), axis=0)
y = np.concatenate((y_prec, y_dry), axis=0)

print(x.shape)
print(y.shape)

idxs = np.arange(x.shape[0])
np.random.shuffle(idxs)
n = x_prec.shape[0] * 2

x = x[idxs,:]
y = y[idxs,:]


print(x.shape)
print(y.shape)
print("AAAAA")
print(x.mean(axis=0))
print(y.mean(axis=0))
print(x.min(axis=0))
print(y.min(axis=0))
print(x.max(axis=0))
print(y.max(axis=0))
print("-------------")

print("-----0.5--------")
p_mask = np.nonzero(y>.5)
x_p = x[p_mask[0], :]
y_p = y[p_mask[0], :]
print(x_p.mean(axis=0))
print(y_p.mean(axis=0))
print(x_p.min(axis=0))
print(y_p.min(axis=0))
print(x_p.max(axis=0))
print(y_p.max(axis=0))

print("-----1.0--------")
p_mask = np.nonzero(y>1.)
x_p = x[p_mask[0], :]
y_p = y[p_mask[0], :]
print(x_p.mean(axis=0))
print(y_p.mean(axis=0))
print(x_p.min(axis=0))
print(y_p.min(axis=0))
print(x_p.max(axis=0))
print(y_p.max(axis=0))

print("-----2.0--------")
p_mask = np.nonzero(y>2.)
x_p = x[p_mask[0], :]
y_p = y[p_mask[0], :]
print(x_p.mean(axis=0))
print(y_p.mean(axis=0))
print(x_p.min(axis=0))
print(y_p.min(axis=0))
print(x_p.max(axis=0))
print(y_p.max(axis=0))


classifier = Sequential()
classifier.add(Dense(32, activation='relu', input_dim=2))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(32, activation='relu'))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(1, activation='relu'))

x_train = x[:750000,:]
x_test = x[750000:,:]
y_train = y[:750000,:]
y_test = y[750000:,:]
print(y_train.shape, y_test.shape)
print(np.square(y_train).mean(axis=0))
print(np.square(y_test).mean(axis=0))
print(np.abs(y_train).mean(axis=0))
print(np.abs(y_test).mean(axis=0))
#classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.compile(optimizer=Adam(lr=0.000001), loss='mae', metrics=['mae', 'mse'])

classifier.fit(x_train, y_train, batch_size=32, nb_epoch=10, validation_data=(x_test, y_test))
