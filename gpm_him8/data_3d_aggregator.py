import xarray as xr 
import numpy as np
import pandas as pd

#dsh = xr.open_mfdataset(["/data/GPM_HIM8/HIM8_201811.nc","/data/GPM_HIM8/HIM8_201901.nc","/data/GPM_HIM8/HIM8_201902.nc"], combine='by_coords')
#dsg = xr.open_mfdataset(["/data/GPM_HIM8/GPM_201811.nc","/data/GPM_HIM8/GPM_201901.nc","/data/GPM_HIM8/GPM_201902.nc"], combine='by_coords')
dsh = xr.open_mfdataset(["/data/GPM_HIM8/HIM8_SYD_201811.nc"], combine='by_coords')
dsg = xr.open_mfdataset(["/data/GPM_HIM8/GPM_201811.nc"], combine='by_coords')

x = None
y = None

block_n = 14

max_t = dsg.time.shape[0]

t_steps = 100
t_i0 = block_n*t_steps

if t_i0 >= max_t:
    exit()

t_span = dsg.time.values[t_i0:t_i0+t_steps]
dr = pd.date_range(start=t_span[0] - np.timedelta64(4*600, 's'), end=t_span[-1], freq='600s')
vals = np.intersect1d(dr.values, dsh.time.values)

b8 = dsh.B8.sel(time=slice(vals[0],vals[-1])).load()
b14 = dsh.B14.sel(time=slice(vals[0],vals[-1])).load()

for t in t_span:
    print(t)
    dr = pd.date_range(end=t, periods=4, freq='600s', closed=None)
    if np.intersect1d(dr.values, dsh.time.values).shape[0] == 4 and t in dsg.time.values:
        if x is None:
            x = np.stack((b8.sel(time=dr).values, b14.sel(time=dr).values), axis=-1)[None,:]

            y = dsg['PrecCal'].sel(time=t).values[None,:,:,None]
            print(x.shape, y.shape)
        else:
            slc = np.stack((b8.sel(time=dr).values, b14.sel(time=dr).values), axis=-1)[None,:]
            x = np.concatenate((x,slc), axis=0)

            slc = dsg['PrecCal'].sel(time=t).values[None,:,:,None]
            y = np.concatenate((y,slc), axis=0)
            print(x.shape, y.shape)

print(x.shape)
exit()

dsg = xr.open_mfdataset(["/data/GPM_HIM8/GPM_201811.nc","/data/GPM_HIM8/GPM_201901.nc","/data/GPM_HIM8/GPM_201902.nc"], combine='by_coords')

dsz = xr.open_mfdataset(["/data/GPM_HIM8/era5_au_z1000_201811.nc","/data/GPM_HIM8/era5_au_z1000_201901.nc","/data/GPM_HIM8/era5_au_z1000_201902.nc"], combine='by_coords')
z1000 = dsz.z.sel(time=dsh.time, method='nearest', tolerance='1800S').values
dsz.close()
z1000 = (z1000-z1000.mean())/z1000.std()

dsz = xr.open_mfdataset(["/data/GPM_HIM8/era5_au_z850_201811.nc","/data/GPM_HIM8/era5_au_z850_201901.nc","/data/GPM_HIM8/era5_au_z850_201902.nc"], combine='by_coords')
z850 = dsz.z.sel(time=dsh.time, method='nearest', tolerance='1800S').values
dsz.close()
z850 = (z850-z850.mean())/z850.std()

dsz = xr.open_mfdataset(["/data/GPM_HIM8/era5_au_z500_201811.nc","/data/GPM_HIM8/era5_au_z500_201901.nc","/data/GPM_HIM8/era5_au_z500_201902.nc"], combine='by_coords')
z500 = dsz.z.sel(time=dsh.time, method='nearest', tolerance='1800S').values
dsz.close()
z500 = (z500-z500.mean())/z500.std()

xz = np.stack((z1000,z850,z500), axis=-1)
z1000 = None
z850 = None
z500 = None

prec = dsg['PrecCal'].sel(time=dsh.time).values
b8 = dsh['B8'].values
b8 = (b8-237)/6.5
b14 = dsh['B14'].values
b14 = (b14-280)/20
x = np.stack((b8,b14), axis=-1)
not_nan = ~np.isnan(x).any(axis=(1,2,3))
x = x[not_nan,:,:,:]
xz = xz[not_nan,:,:,:]
y = prec[not_nan, :, :, None]

b8 = None
b14 = None
prec = None

dsg.close()
dsh.close()


print(x.shape, x.dtype)
print(xz.shape, xz.dtype)
print(y.shape, y.dtype)

idxs = np.arange(x.shape[0])
print(idxs)
np.random.seed(0)
np.random.shuffle(idxs)
print(idxs)
x_train = x[idxs, :, :, :]
xz_train = xz[idxs, :, :, :]
y_train = y[idxs, :]

x = None
xz = None
y = None

print("Train", x_train.shape, xz_train.shape, y_train.shape)

dsh = xr.open_dataset("/data/GPM_HIM8/HIM8_201812.nc")
dsg = xr.open_dataset("/data/GPM_HIM8/GPM_201812.nc")

dsz = xr.open_dataset("/data/GPM_HIM8/era5_au_z1000_201812.nc")
z1000 = dsz.z.sel(time=dsh.time, method='nearest', tolerance='1800S').values
dsz.close()
z1000 = (z1000-z1000.mean())/z1000.std()

dsz = xr.open_dataset("/data/GPM_HIM8/era5_au_z850_201812.nc")
z850 = dsz.z.sel(time=dsh.time, method='nearest', tolerance='1800S').values
dsz.close()
z850 = (z850-z850.mean())/z850.std()

dsz = xr.open_dataset("/data/GPM_HIM8/era5_au_z500_201812.nc")
z500 = dsz.z.sel(time=dsh.time, method='nearest', tolerance='1800S').values
dsz.close()
z500 = (z500-z500.mean())/z500.std()

xz = np.stack((z1000,z850,z500), axis=-1)
z1000 = None
z850 = None
z500 = None
prec = dsg['PrecCal'].sel(time=dsh.time).values
b8 = dsh['B8'].values
b8 = (b8-237)/6.5
b14 = dsh['B14'].values
b14 = (b14-280)/20
x = np.stack((b8,b14), axis=-1)
not_nan = ~np.isnan(x).any(axis=(1,2,3))
x = x[not_nan,:,:,:]
xz = xz[not_nan,:,:,:]
y = prec[not_nan, :, :, None]

b8 = None
b14 = None
prec = None

dsg.close()
dsh.close()

print(x.shape, x.dtype)
print(y.shape, y.dtype)
print(x.shape, x.dtype)
print(y.shape, y.dtype)

x_test = x
xz_test = xz
y_test = y
x = None
xz = None
y = None

print("Test", x_test.shape, xz_test.shape, y_test.shape)

np.save("x_train", x_train)
np.save("xz_train", xz_train)
np.save("y_train", y_train)
np.save("x_test", x_test)
np.save("xz_test", xz_test)
np.save("y_test", y_test)
