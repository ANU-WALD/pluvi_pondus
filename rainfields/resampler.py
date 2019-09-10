import xarray as xr
import numpy as np
from datetime import datetime
from datetime import timedelta

def calculate_log_trans(input_file):
    Rmin = .1
    Zmin = np.log(Rmin)

    ds = xr.open_dataset(input_file)
    ds0 = ds.copy(deep=True)
    ds.close()


    ds0 = ds0.resample(time='30min').mean()
    t_size = ds0.time.data.shape[0]
    cube_flat = ds0.himfields.data.flatten()
    ds0 = ds0.drop(["himfields"])
    
    cube_flat[cube_flat>=Rmin] = np.log(cube_flat[cube_flat>=Rmin]) - Zmin
    cube_flat[cube_flat<Rmin] = 0
    cube = cube_flat.reshape((t_size,2050,2450))

    ds0['himfields'] = (('time', 'y', 'x'), cube)
    ds0.to_netcdf(input_file[:-3] + "_30min.nc".format(start.strftime("%Y%m%d")))
    ds0.close()
    
    
start = datetime(2018, 11, 1)
while start <= datetime(2018, 12, 31):
    print (start.strftime("%Y-%m-%d"))
    f = "/data/pluvi_pondus/Himfields_{}.nc".format(start.strftime("%Y%m%d"))
    calculate_log_trans(f)
    start += timedelta(days=1)
