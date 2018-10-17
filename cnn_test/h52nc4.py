import h5py
import netCDF4
import sys
import os
import argparse
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""GPM image converter""")
    parser.add_argument('-i', '--input', type=str, required=True, help="Date of data.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Date of data.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(1)

    fname = os.path.basename(args.input)
    parts = fname.split("-")
    d = datetime.strptime(parts[2].split(".")[-1]+parts[3][1:], "%Y%m%d%H%M%S")

    with h5py.File(args.input, "r") as f:
        prec = f['//Grid/precipitationCal'][:]
        lon = f['//Grid/lon'][:]
        lat = f['//Grid/lat'][:]

        with netCDF4.Dataset(args.output, 'w', format='NETCDF4_CLASSIC') as ds:
            t_dim = ds.createDimension("time", 1)
            x_dim = ds.createDimension("lon", lon.shape[0])
            y_dim = ds.createDimension("lat", lat.shape[0])

            var = ds.createVariable("time", "f8", ("time",))
            var.units = "seconds since 1970-01-01 00:00:00.0"
            var.calendar = "standard"
            var.long_name = "Time, unix time-stamp"
            var.standard_name = "time"
            var[:] = netCDF4.date2num([d], units="seconds since 1970-01-01 00:00:00.0", calendar="standard")

            var = ds.createVariable("lon", "f8", ("lon",))
            var.units = "degrees_east"
            var.long_name = "longitude"
            var.standard_name = "longitude"
            var.axis = "X"
            var[:] = lon

            var = ds.createVariable("lat", "f8", ("lat",))
            var.units = "degrees_north"
            var.long_name = "latitude"
            var.standard_name = "latitude"
            var.axis = "Y"
            var[:] = lat

            var = ds.createVariable("precipitationCal", 'f4', ("time", "lat", "lon"), fill_value=-9999.9)
            var.long_name = "precipitationCal"
            var.units = 'mm'
            var[:] = prec[None,...]
