import xarray as xr
import numpy as np
from datetime import datetime
from datetime import timedelta

tiles = {"VICTAS": [1130883.7, -4376935], "SYDM": [1740404.3, -3757409], "W_NSW": [935583.1, -3523967], "SA": [189052.7, -3376863], "SE_WA": [-661380.3, -3393506], "SW_WA": [-1414697.2, -3458529], "NW_WA": [-1519737.4, -2568012], "NE_WA": [-710487.2, -2498161], "N_SA": [203089.7, -2480281], "W_QLD": [1010080.4, -2574165], "SE_QLD": [1790401.1, -2828908], "NW_NT": [-704899, -1609515], "NT": [216976.7, -1593277], "N_QLD": [867375.3, -1618745], "NE_QLD": [1597875, -1905570], "NW_WA_COAST": [-1623655.5, -1687006]}

def calculate_tile(input_file, tile_name):

    ds = xr.open_dataset(input_file)

    ic = (np.abs(ds.x.data - tiles[tile_name][0])).argmin()
    jc = (np.abs(ds.y.data - tiles[tile_name][1])).argmin()

    ds = ds.isel(x=slice(ic-255,ic+257), y=slice(jc-255,jc+257))
    ds.to_netcdf(input_file[:-3] + "_{}.nc".format(tile_name))
    ds.close()
    
    
start = datetime(2018, 11, 1)
while start <= datetime(2018, 12, 31):
    print (start.strftime("%Y-%m-%d"))
    f = "/data/pluvi_pondus/Himfields_{}_30min.nc".format(start.strftime("%Y%m%d"))
    for key, _ in tiles.items():
        print(key)
        calculate_tile(f, key)
    start += timedelta(days=1)
