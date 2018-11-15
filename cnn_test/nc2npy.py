from netCDF4 import Dataset
import numpy as np

tp = Dataset("/g/data/fj4/scratch/EU_TP_ERA5.nc", "r")["tp"][:]
print(tp.shape)
np.save("/g/data/fj4/scratch/eu_y", tp.data.astype(np.float32))

z = Dataset("/g/data/fj4/scratch/EU_Z_ERA5.nc", "r")["z"][:, 1, :, :]
print(z.shape)
np.save("/g/data/fj4/scratch/eu_x", z.data.astype(np.float32))
