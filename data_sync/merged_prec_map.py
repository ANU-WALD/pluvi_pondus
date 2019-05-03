import h5py
import xarray as xr
import numpy as np
import datetime
import imageio
import sys
import os
import glob

def get_gpm_precip(fname):
    p = None
    with h5py.File(fname, mode='r') as f:
        p = f['Grid']['precipitationCal'][:].T[::-1, :, 0]
        #p = f['Grid']['precipitationCal'][:].T[::-1, :]
        p[p == -9999.9] = 0
    return p

def get_hres_precip(fname):
    ds = xr.open_dataset(fname)
    
    tp = ds['tp'][0,:,:].data
    cp = ds['cp'][0,:,:].data

    p = (tp - cp) * 1000
    """
    p = np.clip(p, 0, 150)
    p = np.log(1 + p)
    norm_p = np.log(1 + p) / 5.01728
    """
    return p   

if len(sys.argv) != 2:
    sys.exit(1)

date_str = sys.argv[1]
date = datetime.datetime.strptime(date_str, '%Y%m%d')

gpm_imgs = sorted(glob.glob("/g/data/fj4/SatellitePrecip/GPM/global/early/3B-HHR-E.MS.MRG.3IMERG.{}*".format(date_str)))

if len(gpm_imgs) != 48:
    print("The series for the day is not complete") 
    sys.exit(0)

for i in range(8):
    print(i)
    hres_grid = "/g/data/ub8/global/Precipitation/NWP/E1D{}{}{}001.nc".format(datetime.datetime.strftime(date, '%m%d'), "0000", datetime.datetime.strftime(date + datetime.timedelta(hours=3), '%m%d%H'))
    if not os.path.isfile(hres_grid):
        print("HRES file not found") 
        sys.exit(0)

    hres_p = get_hres_precip(hres_grid)
    hres_p = np.repeat(hres_p[:900,:], 2, axis=0)
    hres_p = np.repeat(hres_p, 2, axis=1)
    
    gpm_p = None
    for im in gpm_imgs[i*6:(i+1)*6]:
        p = get_gpm_precip(im)
        if gpm_p is None:
            gpm_p = p
            continue
        gpm_p += p

    p = gpm_p + hres_p
    p = np.clip(p, 0, 150)
    p = np.log(1 + p)
    norm_p = np.log(1 + p) / 5.01728
    im = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.float64)
    im[:,:,2] = 1
    im[:,:,3] = norm_p
    im = (im*255).astype(np.uint8)

    fname = "MRG{}".format(datetime.datetime.strftime(date + datetime.timedelta(hours=(3*(i+1))), "%Y%m%d%H%M"))
    imageio.imwrite("{}.png".format(fname), im)
    os.system("gdal_translate -of GTiff -a_ullr -180 90 180 -90 -a_srs EPSG:4326 {}.png {}.tif".format(fname, fname))
    os.system("gdalwarp -of GTiff -s_srs EPSG:4326 -t_srs EPSG:3857 -te_srs EPSG:4326 -te -180 -85.0511 180 85.0511 {}.tif {}_proj.tif".format(fname, fname))
    os.system("gdal_translate -of PNG {}_proj.tif {}.png".format(fname, fname))
    os.system("rm *.tif")
    os.system("rm *.xml")
