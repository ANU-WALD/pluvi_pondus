import numpy as np
import pandas as pd
import os
import netCDF4
from datetime import datetime
import argparse
import tarfile
import glob

def get_gsod_dataframe(filename):

    col_names = ['STN---', 'WBAN', 'YEARMODA', 'TEMP', 'nTEMP', 'DEWP', 'nDEWP', 'SLP', 'nSLP', 'STP', 'nSTP', 'VISIB', 'nVISIB', 'WDSP', 'nWDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']
    df = pd.read_csv(filename, compression='gzip', header=None, names=col_names, skiprows=1, index_col=2, dtype=object, sep=r'\s{1,}', parse_dates=[2])
    df = df.reindex(pd.date_range('01-01-2015', '31-12-2015'), fill_value=np.NaN)
    #df.index.names = ['time']
    df = df.drop(['STN---', 'WBAN'], axis=1)
    df['TEMP'] = df['TEMP'].replace('9999.9',np.NaN)
    df['TEMP'] = df['TEMP'].astype(np.float32)
    df['nTEMP'] = df['nTEMP'].replace(np.nan, 255).astype(np.uint8)
    df['DEWP'] = df['DEWP'].replace('9999.9',np.NaN)
    df['DEWP'] = df['DEWP'].astype(np.float32)
    df['nDEWP'] = df['nDEWP'].replace(np.nan, 255).astype(np.uint8)
    df['SLP'] = df['SLP'].replace('9999.9',np.NaN)
    df['SLP'] = df['SLP'].astype(np.float32)
    df['nSLP'] = df['nSLP'].replace(np.nan, 255).astype(np.uint8)
    df['STP'] = df['STP'].replace('9999.9',np.NaN)
    df['STP'] = df['STP'].astype(np.float32)
    df['nSTP'] = df['nSTP'].replace(np.nan, 255).astype(np.uint8)
    df['VISIB'] = df['VISIB'].replace('9999.9', np.NaN).replace('999.9',np.NaN)
    df['VISIB'] = df['VISIB'].astype(np.float32)
    df['nVISIB'] = df['nVISIB'].replace(np.nan, 255).astype(np.uint8)
    df['WDSP'] = df['WDSP'].replace('999.9',np.NaN)
    df['WDSP'] = df['WDSP'].astype(np.float32)
    df['nWDSP'] = df['nWDSP'].replace(np.nan, 255).astype(np.uint8)
    df['MXSPD'] = df['MXSPD'].replace('999.9', np.NaN)
    df['MXSPD'] = df['MXSPD'].astype(np.float32)
    df['GUST'] = df['GUST'].replace('999.9', np.NaN)
    df['GUST'] = df['GUST'].astype(np.float32)
    df['MAX'] = df['MAX'].replace('9999.9', np.NaN)
    df['MAX'] = df['MAX'].map(lambda x: str(x).rstrip('*'))
    df['MAX'] = df['MAX'].astype(np.float32)
    df['MIN'] = df['MIN'].replace('9999.9', np.NaN)
    df['MIN'] = df['MIN'].map(lambda x: str(x).rstrip('*'))
    df['MIN'] = df['MIN'].astype(np.float32)
    df['PRCP'] = df['PRCP'].replace('99.99', np.NaN)
    df['tPRCP'] = df['PRCP'].map(lambda x: ord(str(x)[-1])-65)
    df['tPRCP'] = df['tPRCP'].astype(np.uint8)
    df['PRCP'] = df['PRCP'].map(lambda x: str(x)[:-1] if str(x) != "nan" else x)
    df['PRCP'] = df['PRCP'].astype(np.float32)
    df['SNDP'] = df['SNDP'].replace('999.9',np.NaN)
    df['SNDP'] = df['SNDP'].astype(np.float32)

    return df

def write_netcdf(nc_filename, dfs):
    with netCDF4.Dataset(nc_filename, 'w', format='NETCDF4') as dest:
        t_dim = dest.createDimension("time", 365)
        station_dim = dest.createDimension("station", len(list(dfs.keys())))

        var = dest.createVariable("time", "f8", ("time",))
        var.units = "seconds since 1970-01-01 00:00:00.0"
        var.calendar = "standard"
        var.long_name = "Time, unix time-stamp"
        var.standard_name = "time"
        var[:] = netCDF4.date2num([datetime.fromtimestamp(t // 1000000000) for t in dfs[10100].index.values.tolist()], units="seconds since 1970-01-01 00:00:00.0", calendar="standard")
        
        var = dest.createVariable("station", "i4", ("station",))
        var.long_name = "WMO Station ID"
        var.standard_name = "station"
        var[:] = np.array(list(dfs.keys()))

        var = dest.createVariable("precip", "f4", ("time", "station"), fill_value=np.nan)
        var.long_name = "24h precipitation"
        var.units = 'mm'
        arr = np.zeros((365, len(list(dfs.keys()))))
    
        i=0
        for key, df in dfs.items():
            # 0.01 inches to mm
            arr[:,i]=df['PRCP'].values
            i+=1
    
        var[:] = arr
    
        var = dest.createVariable("t_precip", "i", ("time", "station"), fill_value=255)
        var.long_name = "24h precipitation accumulation mode"
        arr = np.zeros((365, len(list(dfs.keys()))))
    
        i=0
        for key, df in dfs.items():
            arr[:,i]=df['tPRCP'].values
            i+=1
    
        var[:] = arr
    
    
        var = dest.createVariable("mean_temp", "f4", ("time", "station"), fill_value=np.nan)
        var.long_name = "24h mean temperature"
        var.units = 'F'
        arr = np.zeros((365, len(list(dfs.keys()))))
    
        i=0
        for key, df in dfs.items():
            # Fahrenheit to Kelvin
            arr[:,i]= (df['TEMP'].values + 459.67) * 5./9.
            i+=1
    
        var[:] = arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-in', '--input_filename', help='GSOD input file name', type=str, required=True)
    parser.add_argument('-out', '--output_filename', help='GSOD netcdf4 file name', type=str, required=True)
    args = vars(parser.parse_args())

    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    else:
        files = glob.glob('tmp/*')
        for f in files:
            os.remove(f)

    if (args["input_filename"].endswith(".tar")):
        tar = tarfile.open(args["input_filename"])
        tar.extractall(path='tmp')
        tar.close()
        print "Extracted in Current Directory"


    dfs = {}

    for i, filename in enumerate(os.listdir("tmp/")):
        if filename.endswith(".gz"): 
            if filename[:6] != "999999":
                dfs[int(filename[:6])] = get_gsod_dataframe("tmp/" + filename)

    write_netcdf(args["output_filename"], dfs)
