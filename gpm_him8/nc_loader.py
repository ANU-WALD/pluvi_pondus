import xarray as xr 
import tensorflow as tf
import numpy as np
import datetime

tf.compat.v1.enable_eager_execution()

class gen:
    dsg = xr.open_dataset("/data/GPM_HIM8/GPM_201811.nc")
    dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_201811.nc")

    def __call__(self, fname, mult):
        
        for t in dsg.time:
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
            
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['precCal'].values
            b8 = dsh['B8'].sel(time=t).values
            b14 = dsh['B14'].sel(time=t).values

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])

        dsg.close()


def HIM8_GPM_Dataset(fnames, mult, batch_size=8):

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32), (tf.TensorShape([512, 512, 2]), tf.TensorShape([128, 128, 1])), args=(fname, mult)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    ds = ds.shuffle(128, seed=None)

    return ds.batch(batch_size)
