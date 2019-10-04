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
            d = datetime.datetime.utcfromtimestamp(t.astype(int) * 1e-9)
            if not os.path.isfile("/home/lar116/project/pablo/rainfields_data/310_{}.prcp-c10.npy".format(d.strftime("%Y%m%d_%H%M%S"))):
                continue
            dp = d - datetime.timedelta(0,10*60)
            if np.datetime64(d) not in dsg.time.data or np.datetime64(dp) not in dsg.time.data:
                continue

            rf_fp = "/home/lar116/project/pablo/rainfields_data/310_{}.prcp-c10.npy".format(d.strftime("%Y%m%d_%H%M%S"))
            prec = np.load(rf_fp)[2::2, 402::2]

            b8 = dsg['B8'].sel(time=t).data[2::2, 402::2]
            b14 = dsg['B14'].sel(time=t).data[2::2, 402::2]
            b8p = dsg['B8'].sel(time=dp).data[2::2, 402::2]
            b14p = dsg['B14'].sel(time=dp).data[2::2, 402::2]

            yield (np.stack((b8,b14,b8p,b14p), axis=-1), prec[:, :, None])

        dsg.close()


def HimfieldsDataset(fnames, mult, batch_size=2):

    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.interleave(lambda fname: tf.data.Dataset.from_generator(gen(), (tf.float32, tf.float32), (tf.TensorShape([1024, 1024, 4]), tf.TensorShape([1024, 1024, 1])), args=(fname, mult)), cycle_length=len(fnames), block_length=1, num_parallel_calls=None)
    #ds = ds.shuffle(128, seed=0)

    return ds.batch(batch_size)
