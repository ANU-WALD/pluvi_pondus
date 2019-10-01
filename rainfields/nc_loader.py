import xarray as xr
import tensorflow as tf
import numpy as np
import datetime
import os

tf.compat.v1.enable_eager_execution()

class gen:
    def __call__(self, fname, mult):
        dsg = xr.open_dataset(fname.decode("utf-8"))
        for t in dsg.time:
            dt = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
            if not os.path.isfile("/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(dt.strftime("%Y%m%d_%H%M%S"))):
                continue
            dsp = xr.open_dataset("/data/pluvi_pondus/Rainfields/310_{}.prcp-c10.nc".format(dt.strftime("%Y%m%d_%H%M%S")))
            prec = dsp['precipitation'].data[2:, 402:] * mult
            dsp.close()
            b8 = dsg['B8'].sel(time=t).data[2:, 402:]
            b14 = dsg['B14'].sel(time=t).data[2:, 402:]

            yield (np.stack((b8,b14), axis=-1), prec[:, :, None])

        dsg.close()


def HimfieldsDataset(fnames, mult, batch_size=2):

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32,tf.float32), (tf.TensorShape([2048, 2048, 2]),tf.TensorShape([2048, 2048, 1])), args=(fname,mult)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    #ds = ds.shuffle(128, seed=0)

    return ds.batch(batch_size)
