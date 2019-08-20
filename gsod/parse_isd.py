import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def get_prec(station_code, date_from):
    col_names = ["year", "month", "day", "hour", "air temp", "dew point", "mslp", "wind dir", "wind speed", "sky cov", "1h prec", "6h prec"]
    df = pd.read_fwf('{}-2019.gz'.format(station_code), compression='gzip', header=None, names=col_names, parse_dates={'datetime': ['year', 'month', 'day', 'hour']})
    df["sky cov"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["6h prec"].replace(to_replace={-9999: np.nan}, inplace=True)
    df["1h prec"].replace(to_replace={-9999: np.nan}, inplace=True)

    for dt in daterange(datetime(2019, 1, 1), datetime(2019, 2, 1)):
        print(dt)
        print(df.loc[(df['datetime'] >= dt) & (df['datetime'] < dt + timedelta(1))]["6h prec"].values)
        print(df.loc[(df['datetime'] >= dt) & (df['datetime'] < dt + timedelta(1))]["1h prec"].values)

    print(np.nansum(df["1h prec"].values))
    print(np.nansum(df["6h prec"].values))
    print(np.nanmax(df["1h prec"].values))
    print(np.nanmax(df["6h prec"].values))
    plt.bar(np.arange(len(df.index))[:450], df["6h prec"].values[:450])
    plt.savefig("6h.png")
    plt.clf() 
    plt.bar(np.arange(len(df.index))[:450], df["1h prec"].values[:450])
    plt.savefig("1h.png")
    return np.nansum(df["1h prec"].values)

get_prec("083300-99999", datetime(2019, 1, 1))
