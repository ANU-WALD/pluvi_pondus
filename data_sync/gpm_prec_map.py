import h5py
import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) != 3:
    sys.exit(1)

def get_precip(fname):
    p = None
    with h5py.File(sys.argv[1], mode='r') as f:
        p = f['Grid']['precipitationCal'][:].T[::-1, :]
        p[p == -9999.9] = 0
    return p

p = get_precip(sys.argv[1])
print(np.nanmin(p), np.nanmean(p), np.nanmax(p))

p = np.log(1 + p)
norm = plt.Normalize()
norm_p = norm(p)

im = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.float64)
im[:,:,2] = 1
im[:,:,3] = norm_p
im = (im*255).astype(np.uint8)
fname = sys.argv[2]
imageio.imwrite("{}.png".format(fname), im)
os.system("gdal_translate -of GTiff -a_ullr -180 90 180 -90 -a_srs EPSG:4326 {}.png {}.tif".format(fname, fname))
os.system("gdalwarp -of GTiff -s_srs EPSG:4326 -t_srs EPSG:3857 -te_srs EPSG:4326 -te -180 -85.0511 180 85.0511 {}.tif {}_proj.tif".format(fname, fname))
os.system("gdal_translate -of PNG {}_proj.tif {}.png".format(fname, fname))
os.system("rm *.tif")
