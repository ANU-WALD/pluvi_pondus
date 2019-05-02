import xarray as xr
import numpy as np
import sys
import imageio
import os

if len(sys.argv) != 3:
    sys.exit(1)

ds = xr.open_dataset(sys.argv[1])

print(ds['tp'].shape)
p = ds['tp'][0,:,:].data * 1000
p = np.clip(p, 0, 150)
p = np.log(1 + p)
norm_p = np.log(1 + p) / 5.01728

im = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.float64)
im[:,:,2] = 1
im[:,:,3] = norm_p
im = (im*255).astype(np.uint8)
fname, _ = os.path.splitext(sys.argv[2])
imageio.imwrite(sys.argv[2], im)
os.system("gdal_translate -of GTiff -a_ullr -180 90 180 -90 -a_srs EPSG:4326 {}.png {}.tif".format(fname, fname))
os.system("gdalwarp -of GTiff -s_srs EPSG:4326 -t_srs EPSG:3857 -te_srs EPSG:4326 -te -180 -85.0511 180 85.0511 {}.tif {}_proj.tif".format(fname, fname))
os.system("gdal_translate -of PNG {}_proj.tif {}.png".format(fname, fname))
os.system("rm *.tif")

print(ds['cp'].shape)
p = ds['cp'][0,:,:].data * 1000
p = np.clip(p, 0, 150)
p = np.log(1 + p)
norm_p = np.log(1 + p) / 5.01728

im = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.float64)
im[:,:,2] = 1
im[:,:,3] = norm_p
im = (im*255).astype(np.uint8)
fname = "CP-" + fname 
imageio.imwrite("{}.png".format(fname), im)
os.system("gdal_translate -of GTiff -a_ullr -180 90 180 -90 -a_srs EPSG:4326 {}.png {}.tif".format(fname, fname))
os.system("gdalwarp -of GTiff -s_srs EPSG:4326 -t_srs EPSG:3857 -te_srs EPSG:4326 -te -180 -85.0511 180 85.0511 {}.tif {}_proj.tif".format(fname, fname))
os.system("gdal_translate -of PNG {}_proj.tif {}.png".format(fname, fname))
os.system("rm *.tif")
