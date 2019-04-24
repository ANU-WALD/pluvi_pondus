import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import mapping
import json
import datetime
import os
import sys
from scipy.interpolate import griddata
from math import pow
from math import sqrt

def pointValue(x, y, power, smoothing, xv, yv, values):
    nominator=0
    denominator=0
    for i in range(0,len(values)):
        dist = sqrt((x-xv[i])*(x-xv[i])+(y-yv[i])*(y-yv[i])+smoothing*smoothing);
        #If the point is really close to one of the data points, return the data point value to avoid singularities
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))
    #Return NODATA if the denominator is zero
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def invDist(xv, yv, values, xsize=100, ysize=100, power=2, smoothing=0):
    valuesGrid = np.zeros((ysize,xsize))
    for x in range(0,xsize):
        for y in range(0,ysize):
            valuesGrid[y][x] = pointValue(x,y,power,smoothing,xv,yv,values)
    return valuesGrid
    

#ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt

def get_prec(station_code, date_from):
    col_names = ["year", "month", "day", "hour", "air temp", "dew point", "mslp", "wind dir", "wind speed", "sky cov", "1h prec", "6h prec"]
    df = None
    try:
        df = pd.read_fwf('{}-2019.gz'.format(station_code), compression='gzip', header=None, names=col_names, parse_dates={'datetime': ['year', 'month', 'day', 'hour']})
    except FileNotFoundError:
        print('File not found.')
        return df

    df["sky cov"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-1: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-1: np.nan}, inplace=True)
    
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
    #min_x, min_y, max_x, max_y = points.bounds
    min_x, min_y, max_x, max_y = 105.0, -46.0, 161.0, -8.0
    min_x = np.floor(min_x)
    min_y = np.floor(min_y)
    max_x = np.ceil(max_x)
    max_y = np.ceil(max_y)
    x_size = int((max_x - min_x) / grid_size)
    y_size = int((max_y - min_y) / grid_size)
    print(x_size, y_size)
    #canvas = np.empty((y_size, x_size), dtype=np.float32)
    canvas = np.zeros((y_size, x_size), dtype=np.float32)

    positions = []
    xv = []
    yv = []
    values = []
    for i, station in ctry_stations.iterrows():
        i, j = int(np.floor((points[i].x - min_x)/grid_size)), int(np.floor((max_y - points[i].y)/grid_size))
        if i < 0 or i >= x_size or j < 0 or j >= y_size:
            continue
        print(j, i)
        #prec = get_prec("{}-{}".format(station["USAF"], station["WBAN"]), date_from)
        prec = get_prec("{}-{}".format(station["USAF"], station["WBAN"]), date_from)
        if prec is None:
            continue
        print(prec)
        print(canvas.shape)
        canvas[j, i] = math.log(1+prec)

        positions.append((i,j))
        xv.append(i)
        yv.append(j)
        values.append(math.log(1+prec))


    #Creating some data, with each coodinate and the values stored in separated lists
    
    #Creating the output grid (100x100, in the example)
    x = np.arange(x_size, dtype=np.int16)
    y = np.arange(y_size, dtype=np.int16)
    XI, YI = np.meshgrid(x,y)

    #Creating the interpolation function and populating the output matrix value
    power=1
    smoothing=10
    ZI = invDist(xv,yv,values,x_size,y_size,power,smoothing)


    # Plotting the result
    plt.subplot(1, 1, 1)
    plt.pcolor(XI, YI, ZI)
    plt.title('Inv dist interpolation - power: ' + str(power) + ' smoothing: ' + str(smoothing))
    plt.colorbar()

    plt.savefig("precip3.png")




    """
    x = np.arange(x_size, dtype=np.int16)
    y = np.arange(y_size, dtype=np.int16)
    X, Y = np.meshgrid(x,y)

    Ti = griddata(positions, values, (X, Y), method='cubic')
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Ti)
    cbar = fig.colorbar(cs)
    plt.savefig("precip2.png")


  
    #print(canvas) 
    plt.imsave("precip.png", canvas) 
    """
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
        if os.path.isfile("{code}-{year}.gz".format(code=station_code, year=date_from.year)):
            continue 
        if os.system('curl -o {code}-{year}.gz ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/{year}/{code}-{year}.gz'.format(code=station_code, year=date_from.year)) != 0:
            print("Error downloading station code", station_code, "for year:", date_from.year)
            continue

if __name__ == "__main__":
    d = datetime.datetime(2019, 3, 1)
    #download_data("Australia", d)
    #update_station_list()
    #update_country_list()
    #lookup_stations("Spain", d)
    raster_stations("Australia", d, .1)
