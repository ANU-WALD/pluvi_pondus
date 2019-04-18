import h5py
import numpy as np
import imageio
import sys
import os
import glob

if len(sys.argv) != 2:
    sys.exit(1)

date = sys.argv[1]
date = "20190416"

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
    imageio.imwrite("GPM3H{}{:02d}.png".format(date, (i+1)*3), im)
