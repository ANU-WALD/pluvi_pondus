import pandas as pd
import os

#ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt

def update_station_list():
    if os.system('curl -o isd-history.csv ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv') != 0:
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
    #if os.system('curl -o country-list.txt ftp://ftp.ncdc.noaa.gov/pub/data/noaa/country-list.txt') != 0:
        #sys.exit(-1)

    df = pd.read_fwf('country-list.txt')
    df.drop([0], inplace=True)
     
    df.to_hdf('country-list.hdf', key='countries', mode='w')
    os.system('rm country-list.txt')

def lookup_station(name, country=None):
    print(df[df['ICAO'].str.contains("YSCB", na=False)])
    # Select stations in Australia
    #df = df.loc[(df['CTRY'] == 'AS') & (df['END'] > 20190101)]
    print(df[df['STATION NAME'].str.contains("BILBAO", na=False)])

if __name__ == "__main__":
    #update_station_list()
    #update_country_list()

