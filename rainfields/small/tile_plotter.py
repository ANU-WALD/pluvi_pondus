import numpy as np 
import imageio
from datetime import datetime
from datetime import timedelta
import os
import xarray as xr

###################################### ColorMap ##############################################
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

color_list = np.array([(255, 255, 255),  # 0.0
                       (245, 245, 255),  # 0.2
                       (180, 180, 255),  # 0.5
                       (120, 120, 255),  # 1.5
                       (20,  20, 255),   # 2.5
                       (0, 216, 195),    # 4.0
                       (0, 150, 144),    # 6.0
                       (0, 102, 102),    # 10
                       (255, 255,   0),  # 15
                       (255, 200,   0),  # 20
                       (255, 150,   0),  # 30
                       (255, 100,   0),  # 40
                       (255,   0,   0),  # 50
                       (200,   0,   0),  # 60
                       (120,   0,   0),  # 75
                       (40,   0,   0)])  # > 100

color_list = color_list/255.
cm = LinearSegmentedColormap.from_list("BOM-RF3", color_list, N=32)

#####################################################################################################################

ds = xr.open_dataset("syd_tile.nc")

for t in ds.time:
  prec = ds.precipitation.sel(time=t).values
  plt.imsave("himfields_syd_{}.png".format(datetime.utcfromtimestamp(t.astype(int)*1e-9).strftime("%Y%m%dT%H%M00")), np.clip(prec,0,5), vmin=0, vmax=5, cmap=cm)
