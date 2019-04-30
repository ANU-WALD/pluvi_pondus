import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import os
import sys
import imageio

#ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt
data_path = "ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/2019/"

def update_station_list():
    if os.system('curl -o isd-history.csv ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv') != 0:
        print("Error downloading isd-history.csv file")
        sys.exit(-1)

    os.system('head isd-history.csv')
    df = pd.read_csv('isd-history.csv', parse_dates=["BEGIN", "END"])
    df["USAF"].fillna("", inplace=True)
    df["STATION NAME"].fillna("", inplace=True)
    df["CTRY"].fillna("", inplace=True)
    df["STATE"].fillna("", inplace=True)
    df["ICAO"].fillna("", inplace=True)

    df.to_hdf('isd-history.hdf', key='stations', mode='w')
    os.system('rm isd-history.csv')


def get_prec(station_code, date_start):
    col_names = ["year", "month", "day", "hour", "air temp", "dew point", "mslp", "wind dir", "wind speed", "sky cov", "1h prec", "6h prec"]
    df = None
    try:
        df = pd.read_fwf(data_path + '{}-2019.gz'.format(station_code), compression='gzip', header=None, names=col_names, parse_dates={'datetime': ['year', 'month', 'day', 'hour']})
    except IOError:
        print('File not found.')
        return None

    df["sky cov"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-1: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-1: np.nan}, inplace=True)


    out = np.zeros(4, dtype=np.float32)
    for i in range(4):
        df = df.loc[(df['datetime'] >= date_start + datetime.timedelta(hours=i*6)) & (df['datetime'] < date_start + datetime.timedelta(hours=(i+1)*6))]
        out[i] = max(np.nansum(df["6h prec"].values), np.nansum(df["1h prec"].values))

    return out


def get_stations(date_from):
    stations = pd.read_hdf("isd-history.hdf", key='stations')
    stns = stations.loc[stations['END'] > date_from]
    for _, station in stns.iterrows():
        yield ("{}-{:05d}".format(station["USAF"], station["WBAN"]), station["LAT"], station["LON"])

    return


def raster_stations(date_from, grid_size):

    min_x, min_y, max_x, max_y = -180.0, -90.0, 180.0, 90.0
    x_size = int((max_x - min_x) / grid_size)
    y_size = int((max_y - min_y) / grid_size)
    map_prec = np.zeros((4, y_size, x_size), dtype=np.float32)

    for station in get_stations(date_from):
        prec = get_prec(station[0], date_from)
        i, j = int(np.floor((station[2] - min_x)/grid_size))-1, int(np.floor((max_y - station[1])/grid_size))-1
        if 0 >= i > x_size or 0 >= j > y_size:
            continue
        map_prec[:, j, i] = prec

    return map_prec

if __name__ == "__main__":
    date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d')
    print(date)
    
    arr = raster_stations(date - datetime.timedelta(hours=24*3), .1)
    print(arr.max(), arr.min())
    prec_map = np.clip(arr, 0, 2*150)
    prec_map += 1
    prec_map = np.log(prec_map)
    prec_map /= np.log(2*150)
    
    for i in range(4):
        im = np.zeros((prec_map.shape[1], prec_map.shape[2], 4), dtype=np.float32)
        im[:, :, 2] = 1
        im[:, :, 3] = prec_map[i,:,:]
        im *= 255
        im = im.astype(np.uint8)
        imageio.imwrite("ISD{}.png".format((date + datetime.timedelta(hours=(6*(i+1)))).strftime("%Y%m%d%H%M")), im)
