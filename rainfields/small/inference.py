import numpy as np 
from tensorflow.keras.models import load_model
#from tensorkeras import backend as K
import tensorflow as tf
import imageio
from datetime import datetime
from datetime import timedelta
import os
import xarray as xr

###################################### ColorMap ##############################################
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

color_list = np.array([(255, 255, 255),  # 0.0
                       (245, 245, 255),  # 0.2
                       (180, 180, 255),  # 0.5
                       (120, 120, 255),  # 1.5
                       (20,  20, 255),   # 2.5
                       (0, 216, 195),    # 4.0
                       (0, 150, 144),    # 6.0
                       (0, 102, 102),    # 10
                       (255, 255,   0),  # 15
                       (255, 200,   0),  # 20
                       (255, 150,   0),  # 30
                       (255, 100,   0),  # 40
                       (255,   0,   0),  # 50
                       (200,   0,   0),  # 60
                       (120,   0,   0),  # 75
                       (40,   0,   0)])  # > 100

color_list = color_list/255.
cm = LinearSegmentedColormap.from_list("BOM-RF3", color_list, N=32)

#####################################################################################################################



def mse_holes(y_true, y_pred):
    idxs = K.tf.where(K.tf.math.logical_not(K.tf.math.is_nan(y_true)))
    y_true = K.tf.gather_nd(y_true, idxs)
    y_pred = K.tf.gather_nd(y_pred, idxs)

    return K.mean(K.square(y_true-y_pred), axis=-1)

#model = load_model('gan_generator.h5')
model = load_model('unet.h5')

d = datetime(2018,11,1,0,10)
i = 0
for index in range(6*24*6):
    print(index, d)
    dp = d - timedelta(0,10*60)
    rf_fp = "/data/pluvi_pondus/Rainfields/310_{}_{}.prcp-c10.nc".format(d.strftime("%Y%m%d"), d.strftime("%H%M%S"))
    h8_fp = "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_{}.nc".format(d.strftime("%Y%m%d"))
    h8p_fp = "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_{}.nc".format(dp.strftime("%Y%m%d"))

    print(rf_fp)
    print(h8_fp)

    if not os.path.exists(rf_fp) or not os.path.exists(h8_fp) or not os.path.exists(h8p_fp):
        d += timedelta(0,10*60)
        print("?", rf_fp)
        print("?", h8_fp)
        continue
           
    rf_ds = xr.open_dataset(rf_fp)
    h8_ds = xr.open_dataset(h8_fp)
    h8p_ds = xr.open_dataset(h8p_fp)
            
    if np.datetime64(d) not in h8_ds.time.data or np.datetime64(dp) not in h8p_ds.time.data:
        d += timedelta(0,10*60)
        continue
           
    prec = rf_ds.precipitation.data[2::2, 402::2]
    b8 = h8_ds.B8.sel(time=d).data[2::2, 402::2]
    b14 = h8_ds.B14.sel(time=d).data[2::2, 402::2]
    b8p = h8p_ds.B8.sel(time=dp).data[2::2, 402::2]
    b14p =h8p_ds.B14.sel(time=dp).data[2::2, 402::2]

    rf_ds.close()
    h8_ds.close()
    h8p_ds.close()
    
    print("Rainfieds: ", np.nanmax(prec))
   
    x = np.stack((b8p,b14p,b8,b14), axis=-1)
    print(x.shape, x.min(), x.max(), x[0,0,0]) 
    #imageio.imwrite("h8_b8_{:03d}.png".format(i), x[:,:,0])
    #plt.imsave("h8_b8_{:03d}.png".format(i), x[2:,402:,0], cmap='gray')
    #imageio.imwrite("rainfields_{:03d}.png".format(i), np.clip(prec, 0, 5)/5)
    #plt.imsave("rainfields_{:03d}.png".format(i), np.clip(prec,0,10), vmin=0, vmax=10, cmap=cm)

    out = model.predict(x[None,:,:,:])
    print("NN: ", out.max())
    #imageio.imwrite("forecasted_{:03d}.png".format(i), np.clip(out[0,:,:,0], 0, 3)/3)
    plt.imsave("forecast_mse_{:03d}.png".format(i), np.clip(out[0,:,:,0],0,10), vmin=0, vmax=10, cmap=cm)

    d += timedelta(0,10*60)
    i+=1

