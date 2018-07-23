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


def get_accum_prec(start, end):
    timestamps = np.arange(start, end, timedelta(hours=1)).astype(datetime)
    accum = np.zeros((721, 1440))

    print("AAAAAA", start, end)
    for t in timestamps:
        run, lead = get_run_lead(t)
        f_path, ti = get_file_index(run, lead)
        with netCDF4.Dataset(f_path, 'r', format='NETCDF4') as dest:
            print(t, dest["tp"][ti, :, :].max(), dest["tp"][ti, :, :].min())
            accum += dest["tp"][ti, :, :]

    return accum


def get_temporal_accum(start, end, accum_h):
    timestamps = np.arange(start, end, timedelta(hours=accum_h)).astype(datetime)

    prec_arr = np.empty((timestamps.shape[0], 721, 1440), dtype=np.float32)

    for i, t in enumerate(timestamps):
        acc = get_accum_prec(t + timedelta(hours=1), t + timedelta(hours=1+accum_h))
        acc[accum<0.001] = 0
        prec_arr[i, :, :] = (acc*1000).astype(np.float32)

    return prec_arr


def get_month_range(year, month):

    start_date = datetime(year, month, 1, 0, 0)
    last_day_month = start_date + timedelta(days=33)
    last_day_month = last_day_month.replace(day=1)
    last_day_month = last_day_month - timedelta(hours=3)

    return start_date, last_day_month


def aggregate_era5(year, month, accum_h):

    with netCDF4.Dataset("/g/data/fj4/ECMWF/ERA5/prec_3h_accum_{}{:02d}.nc".format(year, month), 'w', format='NETCDF4') as dest:

        start, end = get_month_range(2016, i)
        timestamps = np.arange(start, end, timedelta(hours=accum_h)).astype(datetime)
        prec_arr = get_temporal_accum(start, end, accum_h)

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
        var[:] = np.linspace(0, 359.75, 1440)

        var = dest.createVariable("latitude", "f8", ("latitude",))
        var.units = "degrees_north"
        var.long_name = "latitude"
        var[:] = np.linspace(-90, 90, 721)

        var = dest.createVariable("tp", "f4", ("time", "latitude", "longitude"), fill_value=0, zlib=True, chunksizes=(8, 400, 400))
        var.long_name = "Total Precipitation"
        var.units = 'mm'
        var[:] = prec_arr


for i in range(1, 13):
    aggregate_era5(2016, i, 3)
