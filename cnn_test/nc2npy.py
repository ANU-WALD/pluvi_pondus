import xarray as xr
import numpy as np

"""
tp = xr.open_dataset("/g/data/fj4/scratch/AU_NATIVE_TP_ERA5.nc").tp[:].data.astype(np.float32)
print(tp.shape)
np.save("/g/data/fj4/scratch/tp_era5_au", tp[:, :, :])
tp = None

z = xr.open_dataset("/g/data/fj4/scratch/AU_NATIVE_Z_ERA5.nc").z[:].data.astype(np.float32)
print(z.shape)
np.save("/g/data/fj4/scratch/z_era5_au", z[:, :, :, :])
z = None

z = xr.open_dataset("/g/data/fj4/scratch/AU_NATIVE_T_ERA5.nc").t[:].data.astype(np.float32)
print(z.shape)
np.save("/g/data/fj4/scratch/t_era5_au", z[:, :, :, :])
z = None
"""

z = xr.open_dataset("/g/data/fj4/scratch/AU_NATIVE_RH_ERA5.nc").r[:].data.astype(np.float32)
print(z.shape)
np.save("/g/data/fj4/scratch/rh_era5_au", z[:, :, :, :])
z = None

