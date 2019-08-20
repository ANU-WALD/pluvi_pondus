import netCDF4
import h5py
import numpy as np
from datetime import datetime
from datetime import timedelta

def get_gpm_filepath(end):
    pattern = "/g/data/fj4/SatellitePrecip/GPM/global/final/{ym}/3B-HHR.MS.MRG.3IMERG.{ymd}-S{sHMS}-E{eHMS}.{mins_day:04d}.V05B.HDF5"
    start = end - timedelta(minutes=30)
    
    return pattern.format(ym=start.strftime('%Y%m'), ymd=start.strftime('%Y%m%d'), sHMS=start.strftime('%H%M%S'), eHMS=(start+timedelta(minutes=29, seconds=59)).strftime('%H%M%S'), mins_day=start.hour*60+start.minute)


def get_accum_prec(end, accum_h):
    timestamps = np.arange(end - timedelta(hours=accum_h), end, timedelta(hours=1)).astype(datetime) + timedelta(hours=1)
    accum = np.zeros((1800, 3600), dtype=np.float)

    for t in timestamps:
        f_path = get_gpm_filepath(t)
        with h5py.File(f_path, mode='r') as f:
            prec = f['Grid']['precipitationCal'][:].T[::-1, :]
            prec[prec == -9999.9] = np.nan
            accum = accum + prec

    return accum


def get_temporal_accum(timestamps, accum_h):
    prec_arr = np.empty((timestamps.shape[0], 1800, 3600), dtype=np.float32)

    for i, t in enumerate(timestamps):
        acc = get_accum_prec(t, accum_h)
        prec_arr[i, :, :] = acc / 2

    return prec_arr


def get_month_range(year, month, accum_h):
    start_date = datetime(year, month, 1, 3, 0)
    end_date = start_date + timedelta(days=33)
    end_date = end_date.replace(day=1)

    return np.arange(start_date, end_date, timedelta(hours=accum_h)).astype(datetime)


def aggregate_gpm(year, month, accum_h):
    with netCDF4.Dataset("/g/data/fj4/SatellitePrecip/GPM/global/final/prec_{}h_accum_{}{:02d}.nc".format(accum_h, year, month), 'w', format='NETCDF4') as dest:
        timestamps = get_month_range(year, month, accum_h)
        prec_accum = get_temporal_accum(timestamps, accum_h)

        setattr(dest, "date_created", datetime.now().strftime("%Y%m%dT%H%M%S"))
        setattr(dest, "Conventions", "CF-1.6")

        x_dim = dest.createDimension("longitude", prec_accum.shape[2])
        y_dim = dest.createDimension("latitude", prec_accum.shape[1])
        t_dim = dest.createDimension("time", prec_accum.shape[0])

        var = dest.createVariable("time", "f8", ("time",))
        var.units = "seconds since 1900-01-01 00:00:00.0"
        var.calendar = "standard"
        var.long_name = "Time, unix time-stamp"
        var.standard_name = "time"
        var[:] = netCDF4.date2num(timestamps, units="seconds since 1900-01-01 00:00:00.0", calendar="standard")

        var = dest.createVariable("longitude", "f8", ("longitude",))
        var.units = "degrees_east"
        var.long_name = "longitude"
        var[:] = np.linspace(-179.95, 179.95, 3600)

        var = dest.createVariable("latitude", "f8", ("latitude",))
        var.units = "degrees_north"
        var.long_name = "latitude"
        var[:] = np.linspace(89.95, -89.95, 1800)

        var = dest.createVariable("precipitationCal", "f4", ("time", "latitude", "longitude"), fill_value=-9999.9, zlib=True, chunksizes=(8, 400, 400))
        var.long_name = "Precipitation Calibrated"
        var.units = 'mm'
        var[:] = prec_accum

        
from dateutil.relativedelta import relativedelta

for i in range(1, 13):
    aggregate_gpm(2017, i, 1)
