import xarray as xr
import numpy as np
import sys
import imageio

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
imageio.imwrite(sys.argv[2], im)

print(ds['cp'].shape)
p = ds['cp'][0,:,:].data * 1000
p = np.clip(p, 0, 150)
p = np.log(1 + p)
norm_p = np.log(1 + p) / 5.01728

im = np.zeros((p.shape[0], p.shape[1], 4), dtype=np.float64)
im[:,:,2] = 1
im[:,:,3] = norm_p
im = (im*255).astype(np.uint8)
imageio.imwrite("CP-"+sys.argv[2], im)
