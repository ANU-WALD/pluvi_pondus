import netCDF4
import h5py
import numpy as np
from datetime import datetime
from datetime import timedelta

def get_gpm_filepaths(start, accum_steps):
    pattern = "/g/data/fj4/SatellitePrecip/GPM/global/final/{ym}/3B-HHR.MS.MRG.3IMERG.{ymd}-S{sHMS}-E{eHMS}.{mins_day:04d}.V05B.HDF5"
    
    paths = []
    for _ in range(accum_steps):
        paths.append(pattern.format(ym=start.strftime('%Y%m'), ymd=start.strftime('%Y%m%d'), sHMS=start.strftime('%H%M%S'), eHMS=(start+timedelta(minutes=29, seconds=59)).strftime('%H%M%S'), mins_day=start.hour*60+start.minute))
        start = start + timedelta(minutes=30)

    return paths


def aggregate_gpm(start, end):
    
    with netCDF4.Dataset("/g/data/fj4/SatellitePrecip/GPM/global/final/prec_{}.nc".format(start.strftime('%Y%m')), 'w', format='NETCDF4') as dest:
        timestamps = np.arange(start, end, timedelta(hours=3)).astype(datetime)

        prec_arr = np.empty((timestamps.shape[0], 1800, 3600), dtype=np.float32)
        prec_accum = np.empty((1800, 3600), dtype=np.float32)

        for i, t in enumerate(timestamps):  
            prec_accum.fill(0)
            for filepath in get_gpm_filepaths(t, 6):
                with h5py.File(filepath, mode='r') as f:
                    prec = f['Grid']['precipitationCal'][:].T
                    prec[prec==-9999.9] = np.nan
                    prec_accum = prec_accum + prec

            prec_arr[i, :, :] = prec_accum / 2
            
        print(start)

        setattr(dest, "date_created", datetime.now().strftime("%Y%m%dT%H%M%S"))
        setattr(dest, "Conventions", "CF-1.6")

        x_dim = dest.createDimension("longitude", prec_arr.shape[2])
        y_dim = dest.createDimension("latitude", prec_arr.shape[1])
        t_dim = dest.createDimension("time", prec_arr.shape[0])

        var = dest.createVariable("time", "f8", ("time",))
        var.units = "seconds since 1970-01-01 00:00:00.0"
        var.calendar = "standard"
        var.long_name = "Time, unix time-stamp"
        var.standard_name = "time"
        var[:] = netCDF4.date2num(timestamps, units="seconds since 1970-01-01 00:00:00.0", calendar="standard")

        var = dest.createVariable("longitude", "f8", ("longitude",))
        var.units = "degrees_east"
        var.long_name = "longitude"
        var[:] = np.linspace(-179.95, 179.95, 3600)

        var = dest.createVariable("latitude", "f8", ("latitude",))
        var.units = "degrees_north"
        var.long_name = "latitude"
        var[:] = np.linspace(-89.95, 89.95, 1800)

        var = dest.createVariable("precipitationCal", "f4", ("time", "latitude", "longitude"), fill_value=-9999.9, zlib=True, chunksizes=(8, 400, 400))
        var.long_name = "Precipitation Calibrated"
        var.units = 'mm'
        var[:] = prec_arr

        
from dateutil.relativedelta import relativedelta

for i in range(1, 13):
    start = datetime(2016, i, 1)
    aggregate_gpm(start, start + relativedelta(months=1))
