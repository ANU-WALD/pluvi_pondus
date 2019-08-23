import numpy as np 
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import imageio
from datetime import datetime
from datetime import timedelta
import os

def mse_holes(y_true, y_pred):
    idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    y_true = K.tf.gather_nd(y_true, idxs)
    y_pred = K.tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

model = load_model('rainfields_model.h5', custom_objects={'mse_holes': mse_holes})


d = datetime(2018,11,1,10,0)
i = 0
for _ in range(6*24*6):
    print(d)
    dp = d - timedelta(0,10*60)
    rf_fp = "/data/pluvi_pondus/Rainfields/{}/310_{}_{}00.prcp-c10.npy".format(int(d.strftime("%d")), d.strftime("%Y%m%d"), d.strftime("%H%M"))
    b8_fp = "/data/pluvi_pondus/HIM8_AU_{}_B8_{}.npy".format(d.strftime("%Y%m%d"), d.strftime("%H%M%S"))
    b14_fp = "/data/pluvi_pondus/HIM8_AU_{}_B14_{}.npy".format(d.strftime("%Y%m%d"), d.strftime("%H%M%S"))
    b8p_fp = "/data/pluvi_pondus/HIM8_AU_{}_B8_{}.npy".format(dp.strftime("%Y%m%d"), dp.strftime("%H%M%S"))
    b14p_fp = "/data/pluvi_pondus/HIM8_AU_{}_B14_{}.npy".format(dp.strftime("%Y%m%d"), dp.strftime("%H%M%S"))
            
    if not os.path.exists(rf_fp) or not os.path.exists(b8_fp) or not os.path.exists(b8p_fp):
        d += timedelta(0,10*60)
        continue      

    b8 = np.load(b8_fp)[2::2, 402::2]
    b14 = np.load(b14_fp)[2::2, 402::2]
    b8p = np.load(b8p_fp)[2::2, 402::2]
    b14p = np.load(b14p_fp)[2::2, 402::2]
            
    prec = np.load(rf_fp)[2::2, 402::2]

    x = np.stack((b8p,b14p,b8,b14), axis=-1)
    imageio.imwrite("h8_b8_{:03d}.png".format(i), x[:,:,0])
    imageio.imwrite("rainfields_{:03d}.png".format(i), np.clip(prec, 0, 5)/5)

    out = model.predict(x[None,:,:,:])
    imageio.imwrite("forecasted_{:03d}.png".format(i), np.clip(out[0,:,:,0], 0, 3)/3)
    d += timedelta(0,10*60)
    i+=1

