import h5py
import numpy as np
import imageio
import sys
import os
import glob

if len(sys.argv) != 2:
    sys.exit(1)

date = sys.argv[1]

def get_precip(fname):
    p = None
    with h5py.File(fname, mode='r') as f:
        p = f['Grid']['precipitationCal'][:].T[::-1, :]
        p[p == -9999.9] = 0
    return p

gpm_imgs = sorted(glob.glob("/g/data/fj4/SatellitePrecip/GPM/global/early/3B-HHR-E.MS.MRG.3IMERG.{}*".format(date)))

if len(gpm_imgs) != 48:
    print("The series for the day is not complete") 
    sys.exit(0)

for i in range(8):
    print(i)
    p_total = None
    for im in gpm_imgs[i*6:(i+1)*6]:
        print(im)
        p = get_precip(im)
        if p_total is None:
            p_total = p
            continue
        p_total += p

    p_total /= 2
    print(np.nanmin(p_total), np.nanmean(p_total), np.nanmax(p_total))
    p_total = np.clip(p_total, 0, 150)
    norm_p = np.log(1 + p) / 5.01728
    im = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.float64)
    im[:,:,2] = 1
    im[:,:,3] = norm_p
    im = (im*255).astype(np.uint8)
    fname = "GPM3H{}{:02d}".format(date, (i+1)*3)
    imageio.imwrite("{}.png".format(fname), im)
    os.system("gdal_translate -of GTiff -a_ullr -180 90 180 -90 -a_srs EPSG:4326 {}.png {}.tif".format(fname, fname))
    os.system("gdalwarp -of GTiff -s_srs EPSG:4326 -t_srs EPSG:3857 -te_srs EPSG:4326 -te -180 -85.0511 180 85.0511 {}.tif {}_proj.tif".format(fname, fname))
    os.system("gdal_translate -of PNG {}_proj.tif {}.png".format(fname, fname))
    os.system("rm *.tif")
