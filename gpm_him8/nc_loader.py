import xarray as xr 
import tensorflow as tf
import numpy as np
import datetime

tf.compat.v1.enable_eager_execution()

class gent:
    def __call__(self, fname):
        #gpm_fname = fname.decode("utf-8")
        #him_fname = "/data/GPM_HIM8/HIM8_{}.nc".format(gpm_fname[-9:-3])
        
        
        gpm_fname = "/data/GPM_HIM8/GPM_201901.nc"
        him_fname = "/data/GPM_HIM8/HIM8_201901.nc"
        dsg = xr.open_dataset(gpm_fname)
        dsh = xr.open_dataset(him_fname)
        
        for i, t in enumerate(dsg.time):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            b8 = dsh['B8'].sel(time=t).values
            b8 = (b8-237)/6.5
            b14 = dsh['B14'].sel(time=t).values
            b14 = (b14-280)/20

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])
       
        dsg.close()
        dsh.close()


class gen:
    def __call__(self, fname):
        #gpm_fname = fname.decode("utf-8")
        #him_fname = "/data/GPM_HIM8/HIM8_{}.nc".format(gpm_fname[-9:-3])
        
        
        gpm_fname = "/data/GPM_HIM8/GPM_201811.nc"
        him_fname = "/data/GPM_HIM8/HIM8_201811.nc"
        dsg = xr.open_dataset(gpm_fname)
        dsh = xr.open_dataset(him_fname)
        
        for i, t in enumerate(dsg.time):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            b8 = dsh['B8'].sel(time=t).values
            b8 = (b8-237)/6.5
            b14 = dsh['B14'].sel(time=t).values
            b14 = (b14-280)/20

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])
       
        dsg.close()
        dsh.close()
        
        gpm_fname = "/data/GPM_HIM8/GPM_201812.nc"
        him_fname = "/data/GPM_HIM8/HIM8_201812.nc"
        dsg = xr.open_dataset(gpm_fname)
        dsh = xr.open_dataset(him_fname)
        
        for i, t in enumerate(dsg.time[:180]):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            b8 = dsh['B8'].sel(time=t).values
            b8 = (b8-237)/6.5
            b14 = dsh['B14'].sel(time=t).values
            b14 = (b14-280)/20

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])

        for i, t in enumerate(dsg.time[295:]):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            b8 = dsh['B8'].sel(time=t).values
            b8 = (b8-237)/6.5
            b14 = dsh['B14'].sel(time=t).values
            b14 = (b14-280)/20

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])
       
        dsg.close()
        dsh.close()
        
        return
        
        gpm_fname = "/data/GPM_HIM8/GPM_201901.nc"
        him_fname = "/data/GPM_HIM8/HIM8_201901.nc"
        dsg = xr.open_dataset(gpm_fname)
        dsh = xr.open_dataset(him_fname)
        
        for i, t in enumerate(dsg.time[:400]):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            b8 = dsh['B8'].sel(time=t).values
            b8 = (b8-237)/6.5
            b14 = dsh['B14'].sel(time=t).values
            b14 = (b14-280)/20

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])

        for i, t in enumerate(dsg.time[700:]):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            b8 = dsh['B8'].sel(time=t).values
            b8 = (b8-237)/6.5
            b14 = dsh['B14'].sel(time=t).values
            b14 = (b14-280)/20

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])
       
        dsg.close()
        dsh.close()



def HIM8_GPM_Dataset(date_strs, batch_size=32):
    fnames = ["/data/GPM_HIM8/GPM_{}.nc".format(date_str) for date_str in date_strs]

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32), (tf.TensorShape([512, 512, 2]), tf.TensorShape([128, 128, 1])), args=(fname,)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    ds = ds.shuffle(128, seed=None)
    
    return ds.batch(batch_size)

def HIM8_GPM_Dataset_test(date_strs, batch_size=32):
    fnames = ["/data/GPM_HIM8/GPM_{}.nc".format(date_str) for date_str in date_strs]

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gent(), (tf.float32, tf.float32), (tf.TensorShape([512, 512, 2]), tf.TensorShape([128, 128, 1])), args=(fname,)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    ds = ds.shuffle(128, seed=None)
    
    return ds.batch(batch_size)
