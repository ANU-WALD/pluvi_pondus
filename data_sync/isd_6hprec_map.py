import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import os
import sys
import imageio

#ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt


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


def update_country_list():
    if os.system('curl -o country-list.txt ftp://ftp.ncdc.noaa.gov/pub/data/noaa/country-list.txt') != 0:
        print("Error downloading country-list.txt file")
        sys.exit(-1)

    df = pd.read_fwf('country-list.txt')
    df.drop([0], inplace=True)
     
    df.to_hdf('country-list.hdf', key='countries', mode='w')
    os.system('rm country-list.txt')


def get_prec(station_code, date_from, accum):
    col_names = ["year", "month", "day", "hour", "air temp", "dew point", "mslp", "wind dir", "wind speed", "sky cov", "1h prec", "6h prec"]
    df = None
    try:
        df = pd.read_fwf('{}-2019.gz'.format(station_code), compression='gzip', header=None, names=col_names, parse_dates={'datetime': ['year', 'month', 'day', 'hour']})
    except IOError:
        print('File not found.')
        return None

    df["sky cov"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-1: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-1: np.nan}, inplace=True)

    df = df.loc[(df['datetime'] >= date_from) & (df['datetime'] < date_from + datetime.timedelta(hours=accum))]

    return max(np.nansum(df["6h prec"].values), np.nansum(df["1h prec"].values))

def get_stations(date_from):
    stations = pd.read_hdf("isd-history.hdf", key='stations')
    stns = stations.loc[stations['END'] > date_from]
    for _, station in stns.iterrows():
        yield ("{}-{:05d}".format(station["USAF"], station["WBAN"]), station["LAT"], station["LON"])

    return


def raster_stations(date_from, accum, grid_size):

    min_x, min_y, max_x, max_y = -180.0, -90.0, 180.0, 90.0
    x_size = int((max_x - min_x) / grid_size)
    y_size = int((max_y - min_y) / grid_size)
    print(x_size, y_size)
    map = np.zeros((y_size, x_size), dtype=np.float32)
    print(map.shape)

    for station in get_stations(date_from):
        prec = get_prec(station[0], d, accum)
        i, j = int(np.floor((station[2] - min_x)/grid_size))-1, int(np.floor((max_y - station[1])/grid_size))-1
        if 0 >= i > x_size or 0 >= j > y_size:
            print(i, j)
            continue
        map[j, i] = prec

    return map


if __name__ == "__main__":
    for d in (datetime.datetime(2019, 4, 20) + datetime.timedelta(hours=n) for n in range(0,144,6)):
        arr = raster_stations(d, 6, .1)
        print(arr.max(), arr.min())
        map = np.clip(arr, 0, 2*150)
        norm_p = np.log(1 + map) / np.log(2*150)
        im = np.zeros((map.shape[0], map.shape[1], 4), dtype=np.float32)
        im[:, :, 2] = 1
        im[:, :, 3] = norm_p
        im = (im * 255).astype(np.uint8)
        imageio.imwrite("ISD{}.png".format((d + datetime.timedelta(hours=6)).strftime("%Y%m%d%H%M")), im)
