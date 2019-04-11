import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import mapping
import json
import datetime
import os
import sys

#ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt

def get_prec(station_code, date_from):
    col_names = ["year", "month", "day", "hour", "air temp", "dew point", "mslp", "wind dir", "wind speed", "sky cov", "1h prec", "6h prec"]
    df = pd.read_fwf('{}-2019.gz'.format(station_code), compression='gzip', header=None, names=col_names, parse_dates={'datetime': ['year', 'month', 'day', 'hour']})
    df["sky cov"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    
    df = df.loc[df['datetime'] > date_from]

    return np.nansum(df["6h prec"].values)

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

def raster_stations(country_name, date_from, grid_size):
    countries = pd.read_hdf("country-list.hdf", key='countries')
    stations = pd.read_hdf("isd-history.hdf", key='stations')

    if country_name.upper() not in list(countries["COUNTRY NAME"]):
        print(country_name, "is not in the list of countries")
        sys.exit(-1)

    ctry_code = countries.loc[countries['COUNTRY NAME'] == country_name.upper()]['FIPS'].values[0]
    
    ctry_stations = stations.loc[(stations['CTRY'] == ctry_code) & (stations['END'] > date_from)]
    ctry_stations.reset_index(inplace=True)
    
    points = MultiPoint([Point(xy) for xy in zip(ctry_stations["LON"], ctry_stations["LAT"])])
    print(points.bounds)
    min_x, min_y, max_x, max_y = points.bounds
    min_x = np.floor(min_x)
    min_y = np.floor(min_y)
    max_x = np.ceil(max_x)
    max_y = np.ceil(max_y)
    x_size = int((max_x - min_x) / grid_size)
    y_size = int((max_y - min_y) / grid_size)
    print(x_size, y_size)
    #canvas = np.empty((y_size, x_size), dtype=np.float32)
    canvas = np.zeros((y_size, x_size), dtype=np.float32)

    for i, station in ctry_stations.iterrows():
        i, j = int(np.floor((points[i].x - min_x)/grid_size)), int(np.floor((max_y - points[i].y)/grid_size))
        print(j, i)
        prec = get_prec("{}-{}".format(station["USAF"], station["WBAN"]), date_from)
        prec = get_prec("{}-{}".format(station["USAF"], station["WBAN"]), date_from)
        print(prec)
        print(canvas.shape)
        canvas[j, i] = prec
  
    print(canvas) 
    plt.imsave("precip.png", canvas) 
    return

def lookup_stations(country_name, date_from):
    countries = pd.read_hdf("country-list.hdf", key='countries')
    stations = pd.read_hdf("isd-history.hdf", key='stations')

    if country_name.upper() not in list(countries["COUNTRY NAME"]):
        print(country_name, "is not in the list of countries")
        sys.exit(-1)

    ctry_code = countries.loc[countries['COUNTRY NAME'] == country_name.upper()]['FIPS'].values[0]
    
    ctry_stations = stations.loc[(stations['CTRY'] == ctry_code) & (stations['END'] > date_from)]
    
    return ["{}-{}".format(station["USAF"], station["WBAN"]) for _, station in ctry_stations.iterrows()]
   
    sys.exit() 
    points = MultiPoint([Point(xy) for xy in zip(ctry_stations["LON"], ctry_stations["LAT"])])
    print(json.dumps(mapping(points)))
    return

    print(countries.head())
    print(stations.head())
    print(df[df['ICAO'].str.contains("YSCB", na=False)])
    # Select stations in Australia
    print(df[df['STATION NAME'].str.contains("BILBAO", na=False)])

def download_data(country_name, date_from):
    #ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/2019/782560-99999-2019.gz

    station_codes = lookup_stations(country_name, date_from)
    for station_code in station_codes:
        if os.system('curl -o {code}-{year}.gz ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/{year}/{code}-{year}.gz'.format(code=station_code, year=date_from.year)) != 0:
            print("Error downloading station code", station_code, "for year:", date_form.year)
            sys.exit(-1)

if __name__ == "__main__":
    d = datetime.datetime(2019, 3, 1)
    download_data("Australia", d)
    #update_station_list()
    #update_country_list()
    #lookup_stations("Spain", d)
    raster_stations("Australia", d, .1)
