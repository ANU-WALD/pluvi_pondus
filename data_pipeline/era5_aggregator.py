import netCDF4
import numpy as np
from datetime import datetime
from datetime import timedelta

def get_run_lead(t):

    if 0 <= t.hour <= 6:
        t_run = t - timedelta(hours=24)
        return t_run.replace(hour=18), 6 + t.hour

    elif 7 <= t.hour <= 18:
        t_run = t
        return t_run.replace(hour=6), t.hour - 6

    elif 19 <= t.hour <= 23:
        t_run = t
        return t_run.replace(hour=18), t.hour - 18


def get_file_index(run, lead):
    run_idx = int((run - datetime(run.year, run.month, 1, 7)).total_seconds() / 3600)
    return "/g/data/fj4/ECMWF/ERA5/precip_{}.nc".format(run.strftime('%Y%m')), run_idx + lead


def get_accum_prec(end, accum_h):
    timestamps = np.arange(end - timedelta(hours=accum_h), end, timedelta(hours=1)).astype(datetime) + timedelta(hours=1)
    accum = np.zeros((721, 1440))

    for t in timestamps:
        run, lead = get_run_lead(t)
        f_path, ti = get_file_index(run, lead)
        with netCDF4.Dataset(f_path, 'r', format='NETCDF4') as dest:
            accum += dest["tp"][ti, :, :]

    return accum


def get_temporal_accum(timestamps, accum_h):
    prec_arr = np.empty((timestamps.shape[0], 721, 1440), dtype=np.float32)

    for i, t in enumerate(timestamps):
        acc = get_accum_prec(t, accum_h) * 1000
        acc[acc < 0.01] = 0
        prec_arr[i, :, :] = acc.astype(np.float32)

    return prec_arr


def get_month_range(year, month, accum_h):
    start_date = datetime(year, month, 1, 3, 0)
    end_date = start_date + timedelta(days=33)
    end_date = end_date.replace(day=1)

    return np.arange(start_date, end_date, timedelta(hours=accum_h)).astype(datetime)


def aggregate_era5(year, month, accum_h):

    with netCDF4.Dataset("/g/data/fj4/ECMWF/ERA5/prec_3h_accum_{}{:02d}.nc".format(year, month), 'w', format='NETCDF4') as dest:
        timestamps = get_month_range(year, month, accum_h)
        prec_arr = get_temporal_accum(timestamps, accum_h)

        setattr(dest, "date_created", datetime.now().strftime("%Y%m%dT%H%M%S"))
        setattr(dest, "Conventions", "CF-1.6")

        x_dim = dest.createDimension("longitude", prec_arr.shape[2])
        y_dim = dest.createDimension("latitude", prec_arr.shape[1])
        t_dim = dest.createDimension("time", prec_arr.shape[0])

        var = dest.createVariable("time", "f8", ("time",))
        var.units = "seconds since 1900-01-01 00:00:00.0"
        var.calendar = "standard"
        var.long_name = "Time, unix time-stamp"
        var.standard_name = "time"
        var[:] = netCDF4.date2num(timestamps, units="seconds since 1900-01-01 00:00:00.0", calendar="standard")

        var = dest.createVariable("longitude", "f8", ("longitude",))
        var.units = "degrees_east"
        var.long_name = "longitude"
        #var[:] = np.linspace(0, 359.75, 1440)
        var[:] = np.linspace(-180, 179.75, 1440)

        var = dest.createVariable("latitude", "f8", ("latitude",))
        var.units = "degrees_north"
        var.long_name = "latitude"
        var[:] = np.linspace(90, -90, 721)

        var = dest.createVariable("tp", "f4", ("time", "latitude", "longitude"), fill_value=9999.9, zlib=True, chunksizes=(8, 400, 400))
        var.long_name = "Total Precipitation"
        var.units = 'mm'
        #var[:] = prec_arr
        var[:] = np.roll(prec_arr, 720, axis=2)


for i in range(1, 13):
    aggregate_era5(2017, i, 3)
