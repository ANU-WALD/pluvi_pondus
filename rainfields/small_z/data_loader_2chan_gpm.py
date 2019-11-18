import xarray as xr 
import tensorflow as tf
import numpy as np
import datetime
import os
import random

tf.compat.v1.enable_eager_execution()

class gen:
    def __call__(self, fname, z=False, jitter=False):
        dsg = xr.open_dataset(fname.decode("utf-8"))
        if z:
            dsz1000 = xr.open_dataset("/data/pluvi_pondus/ERA5/au_z1000_201811.nc")
            dsz800 = xr.open_dataset("/data/pluvi_pondus/ERA5/au_z800_201811.nc")
            dsz500 = xr.open_dataset("/data/pluvi_pondus/ERA5/au_z500_201811.nc")

        for t in dsg.time:
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
            
            if not os.path.isfile("/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(d.strftime("%Y%m%d_%H%M%S"))):
                continue
            if np.datetime64(d) not in dsg.time.data:
                continue
            if z:
                z500 = dsz500['z'].sel(time=t, method='nearest').data[:, 50:]
                z800 = dsz800['z'].sel(time=t, method='nearest').data[:, 50:]
                z1000 = dsz1000['z'].sel(time=t, method='nearest').data[:, 50:]

            rf_fp = "/data/pluvi_pondus/GPM/GPM_BoM_201811.nc"
            dsp = xr.open_dataset(rf_fp)
            prec = dsp['PrecCal'].sel(time=t, method='nearest').data[1:, 101:]
            b8 = dsg['B8'].sel(time=t).data[2::2, 402::2]
            b14 = dsg['B14'].sel(time=t).data[2::2, 402::2]
           
            if z:
                yield (np.stack((b8,b14), axis=-1), np.stack((z1000,z800,z500), axis=-1), prec[:, :, None])
            else:
                yield (np.stack((b8,b14), axis=-1), prec[:, :, None])

        dsg.close()


def HimfieldsDataset(fnames, z, batch_size=2):

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    if z:
        ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32, tf.float32), (tf.TensorShape([1024, 1024, 2]), tf.TensorShape([256,256,3]), tf.TensorShape([512, 512, 1])), args=(fname, z)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    else:
        ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32), (tf.TensorShape([1024, 1024, 2]), tf.TensorShape([512, 512, 1])), args=(fname, z)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    ds = ds.shuffle(128, seed=None)
    ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
