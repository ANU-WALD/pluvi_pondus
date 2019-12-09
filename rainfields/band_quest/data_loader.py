import xarray as xr 
import tensorflow as tf
import numpy as np
import datetime
import os
import random

tf.compat.v1.enable_eager_execution()

class gen:
    def __call__(self, fname, band, jitter=False):
        dsg = xr.open_dataset(fname.decode("utf-8"))

        for t in dsg.time:
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
            
            if not os.path.isfile("/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(d.strftime("%Y%m%d_%H%M%S"))):
                continue
            if np.datetime64(d) not in dsg.time.data:
                continue


            rf_fp = "/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(d.strftime("%Y%m%d_%H%M%S"))
            dsp = xr.open_dataset(rf_fp)
            prec = dsp['precipitation'].data[2::2, 402::2]
            b = dsg['B{}'.format(band)].sel(time=t).data[2::2, 402::2]
           
            # Added Normalisation
            b = b / 273
           
            yield (b[:, :, None], prec[:, :, None])

        dsg.close()

def CompleteFNames(fnames, band):
    d = None
    if band in [8, 12]:
        d = '0812'
    elif band in [9, 10]:
        d = '0910'
    elif band in [11, 13]:
        d = '1113'
    elif band in [14, 15]:
        d = '1415'

    return [path.format(d) for path in fnames]


def HimfieldsDataset(fnames, band, batch_size=2):

    fnames = CompleteFNames(fnames, band)
    print(fnames)

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32), (tf.TensorShape([1024, 1024, 1]), tf.TensorShape([1024, 1024, 1])), args=(fname, band)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    
    ds = ds.shuffle(128, seed=None)
    ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
