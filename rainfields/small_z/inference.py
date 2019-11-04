import numpy as np 
from tensorflow.keras.models import load_model
#from tensorkeras import backend as K
import tensorflow as tf
import imageio
from datetime import datetime
from datetime import timedelta
import os
import xarray as xr
from itertools import chain

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

gan = load_model('gan_2chan_generator.h5')
ganrc = load_model('gan_2chanrc_generator.h5')
mse = load_model('unet.h5')

for nd in chain(range(2,3), range(5,6), range(27,28)):
    #stack = None
    print("gan day", nd)
    d = datetime(2018, 11, nd, 0, 0)
    i = 0

    for index in range(6*24):
        dp = d - timedelta(0,10*60)
        rf_fp = "/data/pluvi_pondus/Rainfields/310_{}_{}.prcp-c10.nc".format(d.strftime("%Y%m%d"), d.strftime("%H%M%S"))
        h8_fp = "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_{}.nc".format(d.strftime("%Y%m%d"))
        h8p_fp = "/data/pluvi_pondus/HIM8_AU_2B/HIM8_2B_AU_{}.nc".format(dp.strftime("%Y%m%d"))

        if not os.path.exists(rf_fp) or not os.path.exists(h8_fp) or not os.path.exists(h8p_fp):
            """
            out = np.empty((1,1025,1225), dtype=np.float32)
            if stack is None:
                stack = out
            else:
                stack = np.concatenate((stack,out), axis=0)
            """
            d += timedelta(0,10*60)
            continue
           
        rf_ds = xr.open_dataset(rf_fp)
        h8_ds = xr.open_dataset(h8_fp)
        h8p_ds = xr.open_dataset(h8p_fp)
            
        if np.datetime64(d) not in h8_ds.time.data or np.datetime64(dp) not in h8p_ds.time.data:
            out = np.empty((1,1025,1225), dtype=np.float32)
            """
            if stack is None:
                stack = out
            else:
                stack = np.concatenate((stack,out), axis=0)
            """
            d += timedelta(0,10*60)
            continue
           
        prec = rf_ds.precipitation.data[::2, ::2]
        
        Z = np.where(prec > .1001, np.log(prec) - np.log(.1), 0)
        R_rec = np.where(Z > 0, np.exp(Z + np.log(.1)), 0)

        plt.imsave("R_Rainfields_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(prec[:,:], 0, 10), vmin=0, vmax=10, cmap=cm)
        plt.imsave("Z_Rainfields_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(Z[:,:], 0, 5), vmin=0, vmax=5, cmap=cm)
        plt.imsave("R_rec_Rainfields_{}.png".format(d.strftime("%Y%m%dT%H%M00")), R_rec, vmin=0, vmax=10, cmap=cm)
        print("P:", np.nanmax(prec))
    
        b8 = h8_ds.B8.sel(time=d).data[::2, ::2]
        b14 = h8_ds.B14.sel(time=d).data[::2, ::2]
        b8p = h8p_ds.B8.sel(time=dp).data[::2, ::2]
        b14p =h8p_ds.B14.sel(time=dp).data[::2, ::2]
    
        rf_ds.close()
        h8_ds.close()
        h8p_ds.close()

        x = np.stack((b8p,b14p,b8,b14), axis=-1)

        out = np.zeros((1025, 1225), dtype=np.float32)[None,:,:]
        out[:,:-1,:-201] = gan.predict(x[None,:-1,:-201,:2])[:,:,:,0]
        out[:,1:,201:] = gan.predict(x[None,1:,201:,:2])[:,:,:,0]
        out[:,-20:,:] = 0
        print("GAN", out.max())
        
        Z = np.where(out > .1, np.log(out) - np.log(.1), 0)
        plt.imsave("R_GAN_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(out[0,:,:], 0, 10), vmin=0, vmax=10, cmap=cm)
        plt.imsave("Z_GAN_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(Z[0,:,:], 0, 5), vmin=0, vmax=5, cmap=cm)

        out = np.zeros((1025, 1225), dtype=np.float32)[None,:,:]
        out[:,:-1,:-201] = ganrc.predict(x[None,:-1,:-201,:2])[:,:,:,0]
        out[:,1:,201:] = ganrc.predict(x[None,1:,201:,:2])[:,:,:,0]
        out[:,-20:,:] = 0
        print("GAN RC", out.max())
        
        Z = np.where(out > .1001, np.log(out) - np.log(.1), 0)
        plt.imsave("R_GAN_RC_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(out[0,:,:], 0, 10), vmin=0, vmax=10, cmap=cm)
        plt.imsave("Z_GAN_RC_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(Z[0,:,:], 0, 5), vmin=0, vmax=5, cmap=cm)
   

        out = np.zeros((1025, 1225), dtype=np.float32)[None,:,:]
        out[:,:-1,:-201] = mse.predict(x[None,:-1,:-201,:])[:,:,:,0]
        out[:,1:,201:] = mse.predict(x[None,1:,201:,:])[:,:,:,0]
        print("MSE", out.max())
        
        Z = np.where(out > .1001, np.log(out) - np.log(.1), 0)
        plt.imsave("R_MSE_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(out[0,:,:], 0, 10), vmin=0, vmax=10, cmap=cm)
        plt.imsave("Z_MSE_{}.png".format(d.strftime("%Y%m%dT%H%M00")), np.clip(Z[0,:,:], 0, 5), vmin=0, vmax=5, cmap=cm)
        
        """
        if stack is None:
            stack = out
        else:
            stack = np.concatenate((stack,out), axis=0)
        """
        #plt.imsave("forecastA_mse_{:03d}.png".format(i), np.clip(a[0,:,:,0],0,10), vmin=0, vmax=10, cmap=cm)

        #print("F", out.max())
        #plt.imsave("forecast_mse_{:03d}.png".format(i), np.clip(out,0,10), vmin=0, vmax=10, cmap=cm)

        #print(x.shape, x.min(), x.max(), x[0,0,0]) 
        #imageio.imwrite("h8_b8_{:03d}.png".format(i), x[:,:,0])
        #plt.imsave("h8_b8_{:03d}.png".format(i), x[2:,402:,0], cmap='gray')
        #imageio.imwrite("rainfields_{:03d}.png".format(i), np.clip(prec, 0, 5)/5)

        #imageio.imwrite("forecasted_{:03d}.png".format(i), np.clip(out[0,:,:,0], 0, 3)/3)
        #plt.imsave("forecast_mse_{:03d}.png".format(i), np.clip(out[0,:,:,0],0,10), vmin=0, vmax=10, cmap=cm)

        d += timedelta(0,10*60)
        i+=1

    #np.save("gan_{}".format(nd), stack)
