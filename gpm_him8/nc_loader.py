import xarray as xr 
import tensorflow as tf
import numpy as np
import datetime

tf.compat.v1.enable_eager_execution()

class gen:

    def __call__(self, date_str):
        dsg = xr.open_dataset("/data/GPM_HIM8/GPM_{}.nc".format(date_str.decode("utf-8")))
        dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_{}.nc".format(date_str.decode("utf-8")))
        
        for i, t in enumerate(dsg.time):
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
           
            if np.datetime64(d) not in dsh.time.values:
                continue

            prec = dsg['PrecCal'].sel(time=t).values
            #b8 = (dsh['B8'].sel(time=t).values-196)#/(256-196)
            b8 = (dsh['B8'].sel(time=t).values)#/(256-196)
            #b14 = (dsh['B14'].sel(time=t).values-196)#/(327-196)
            b14 = (dsh['B14'].sel(time=t).values)#/(327-196)

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])

        dsg.close()


def HIM8_GPM_Dataset(date_str, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(["/data/GPM_HIM8/GPM_{}.nc".format(date_str)])
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32), (tf.TensorShape([512, 512, 2]), tf.TensorShape([128, 128, 1])), args=(date_str,)), cycle_length=1, block_length=1, num_parallel_calls=None)
    ds = ds.shuffle(128, seed=None)

    return ds.batch(batch_size)
